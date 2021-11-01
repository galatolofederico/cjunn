import unittest
import numpy as np
import torch

from src.models.cjunn.model import CombinedJointUNNModel


def run_model_single(model, network, input, return_postsynaptic=False):
    postsynaptic = torch.cat((input, torch.zeros(1), torch.ones(1), torch.zeros(model.hidden+model.outputs)))
    for node in range(0, model.hidden+model.outputs):
        incoming_indices = model.connections[network, node]
        incoming_weights = model.W[incoming_indices, node]
        incoming_values = postsynaptic[incoming_indices]

        outgoing_value = (incoming_values * incoming_weights).sum()
        outgoing_value = model.activation(outgoing_value)

        postsynaptic[model.inputs + node] = outgoing_value
    if return_postsynaptic:
        return postsynaptic
    return postsynaptic[-model.outputs:]




class TestCombinedJointUNNModel(unittest.TestCase):
    def test_forward(self):
        batch_size = 2
        inputs = 3
        hidden = 5
        outputs = 2
        networks = 3
        max_incoming_connections = 10

        model = CombinedJointUNNModel(inputs, hidden, outputs, networks, max_incoming_connections)
        X = torch.randn(batch_size, inputs)
        y = model(X)
        self.assertTrue(np.allclose(y.shape, [batch_size, outputs]))
    
    def test_backward(self):
        batch_size = 2
        inputs = 3
        hidden = 5
        outputs = 2
        networks = 3
        max_incoming_connections = 10

        model = CombinedJointUNNModel(inputs, hidden, outputs, networks, max_incoming_connections)
        X = torch.randn(batch_size, inputs)
        y = model(X, weighted=False)

        y.sum().backward()

        self.assertTrue(hasattr(model.W, "grad"))
        for i in range(0, len(model.networks_weights)):
            self.assertTrue(hasattr(model.networks_weights[i], "grad"))


    def test_gradients(self):
        '''
            w0
        (i0)-------
            w1    |      w4
        (i1)------+--(f)----(f)---
            w2    |        |
        (i2)-------        |
            w3             |
        (i3)----------------

        (b0)

        (b1)

        complete: (i+h+o) x (i+h+o)
        from/to i0 i1 i2 b0 b1 h  o 
        i0      0  0  0  0  0  w0 0
        i1      0  0  0  0  0  w1 0
        i2      0  0  0  0  0  w2 0
        i3      0  0  0  0  0  0  w3
        b0      0  0  0  0  0  0  0
        b1      0  0  0  0  0  0  0
        h       0  0  0  0  0  w4 0
        o       0  0  0  0  0  0  0

        actual: (i+h) x (h+o)
        from/to h  o 
        i0      w0 0
        i1      w1 0
        i2      w2 0
        i3      0  w3
        b0      0  0
        b1      0  0
        h       0  w4


        '''
        i0 = 0.5
        i1 = 0.2
        i2 = 1
        i3 = 0.4

        w0 = 0.3
        w1 = 0.2
        w2 = 0.8
        w3 = 0.6
        w4 = 0.7

        #computed by hand
        w0_grad = 0.015061
        w1_grad = 0.006025
        w2_grad = 0.030123
        w3_grad = 0.087147
        w4_grad = 0.158844

        model = CombinedJointUNNModel(4, 1, 1, 1, 3, activation=torch.sigmoid)
        
        model.W.data.zero_()
        model.W.data[0, 0] = w0
        model.W.data[1, 0] = w1
        model.W.data[2, 0] = w2
        model.W.data[3, 1] = w3
        model.W.data[6, 1] = w4
        
        #connections[to (h+o), from node (i+h)] aka incoming nodes for n is connections[n]
        model.connections[0, 0, :] = torch.tensor([0, 1, 2])
        model.connections[0, 1, :] = torch.tensor([3, 6, 4]) #4 is the zero-bias

        inputs = torch.tensor([[i0, i1, i2, i3]])

        outputs = model(inputs)
        outputs[0, 0].backward()

        self.assertAlmostEqual(model.W.grad[0, 0].item(), w0_grad, places=5)
        self.assertAlmostEqual(model.W.grad[1, 0].item(), w1_grad, places=5)
        self.assertAlmostEqual(model.W.grad[2, 0].item(), w2_grad, places=5)
        self.assertAlmostEqual(model.W.grad[3, 1].item(), w3_grad, places=5)
        self.assertAlmostEqual(model.W.grad[6, 1].item(), w4_grad, places=5)


    def test_batch_model_1(self):
        batch_size = 5
        inputs = 2
        hidden = 10
        outputs = 2
        networks = 3
        max_incoming_connections = 10

        model = CombinedJointUNNModel(inputs, hidden, outputs, networks, max_incoming_connections)

        X = torch.randn(batch_size, inputs)
        serial_outputs = torch.zeros(batch_size, networks, inputs+2+hidden+outputs)
        for b in range(0, batch_size):
            for n in range(0, networks):
                serial_outputs[b, n, :] = run_model_single(model, n, X[b], return_postsynaptic=True)

        #networks_weights = model.networks_weights.unsqueeze(0).unsqueeze(2)
        #serial_outputs = (serial_outputs * networks_weights).sum(dim=1)

        batched_outputs = model(X, return_postsynaptic=True)
        self.assertTrue(np.allclose(serial_outputs.detach().numpy(), batched_outputs.detach().numpy()))

    def test_batch_model_2(self):
        batch_size = 3
        inputs = 3
        hidden = 5
        outputs = 2
        networks = 5
        max_incoming_connections = 10

        model = CombinedJointUNNModel(inputs, hidden, outputs, networks, max_incoming_connections)
        connections = model.connections.clone()
        X = torch.randn(batch_size, inputs)

        serial_outputs_1 = torch.zeros(batch_size, networks, outputs)
        serial_outputs_2 = torch.zeros(batch_size, networks, outputs)
        
        for b in range(0, batch_size):
            for n in range(0, networks):
                model.networks = connections.shape[0]
                model.connections = connections
                serial_outputs_1[b, n, :] = run_model_single(model, n, X[b])

                serial_X = X[b].unsqueeze(0)
                model.networks = 1
                model.connections = connections[n].unsqueeze(0)

                serial_outputs_2[b, n, :] = model(serial_X, weighted=False)
        
        model.networks = connections.shape[0]
        model.connections = connections
        batched_outputs = model(X, weighted=False)

        serial_outputs_1 = serial_outputs_1.detach().numpy()
        serial_outputs_2 = serial_outputs_2.detach().numpy()

        self.assertTrue(np.allclose(serial_outputs_1, serial_outputs_2))



    def test_batch_model_3(self):
        batch_size = 7
        inputs = 3
        hidden = 4
        outputs = 2
        networks = 5
        max_incoming_connections = 15

        model = CombinedJointUNNModel(inputs, hidden, outputs, networks, max_incoming_connections)

        X = torch.randn(batch_size, inputs)
        serial_outputs = torch.zeros(batch_size, networks, outputs)
        for b in range(0, batch_size):
            for n in range(0, networks):
                serial_outputs[b, n, :] = run_model_single(model, n, X[b])

        networks_weights = model.get_networks_weights().unsqueeze(0).unsqueeze(2)
        serial_outputs = (serial_outputs * networks_weights).sum(dim=1)

        batched_outputs = model(X)
        self.assertTrue(np.allclose(serial_outputs.detach().numpy(), batched_outputs.detach().numpy()))


if __name__ == '__main__':
    unittest.main()