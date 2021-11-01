import unittest
import numpy as np
import torch
from opt_einsum import contract

from src.models.cjunn.model import CombinedJointUNNModel


class TestMisc(unittest.TestCase):
    def test_einsum(self):
        batch_size = 5
        inputs = 3
        hidden = 4
        outputs = 2
        networks = 3
        max_incoming_connections = 10

        model = CombinedJointUNNModel(inputs, hidden, outputs, networks, max_incoming_connections)
        X = torch.randn(batch_size, inputs)
        batch_size = X.shape[0]

        X = torch.cat((X, torch.zeros(batch_size, 1), torch.ones(batch_size, 1)), dim=1)
        postsynaptic = torch.cat((X, torch.zeros((batch_size, model.hidden+model.outputs))), dim=1)
        neurons = model.inputs + model.hidden + model.outputs
        postsynaptic = postsynaptic.reshape(batch_size, 1, neurons)
        postsynaptic = postsynaptic.repeat(1, model.networks, 1)

        incoming_values_target = torch.zeros(batch_size, model.networks, model.max_incoming_connections)
        outgoing_values_loop = torch.zeros(batch_size, model.networks)

        for node in range(0, model.hidden+model.outputs):
            incoming_indices = model.connections[:, node] 
            for b in range(0, batch_size):
                for n in range(0, model.networks):
                    total = 0
                    for i in range(0, model.max_incoming_connections):
                        ix = incoming_indices[n, i]
                        incoming_values_target[b, n, i] = postsynaptic[b, n, ix] ##
                        total += postsynaptic[b, n, ix] * model.W[ix, node]
                    outgoing_values_loop[b, n] = total

            incoming_weights = model.W[incoming_indices, node]
            outgoing_values_mul = incoming_weights * incoming_values_target
            outgoing_values_mul = outgoing_values_mul.sum(dim=2)

            incoming_indices_batch = incoming_indices.unsqueeze(0)
            incoming_indices_batch = incoming_indices_batch.repeat(batch_size, 1, 1)
            
            incoming_values = torch.gather(postsynaptic, 2, incoming_indices_batch)
            incoming_weights = model.W[incoming_indices, node]
            outgoing_values = contract("ijk,jk->ij", incoming_values, incoming_weights)

            self.assertTrue(np.allclose(outgoing_values_loop.detach().numpy(), outgoing_values_mul.detach().numpy()))   
            self.assertTrue(np.allclose(outgoing_values.detach().numpy(), outgoing_values_mul.detach().numpy()))   

            postsynaptic[:, :, model.inputs + node] = model.activation(outgoing_values_mul)

            


if __name__ == '__main__':
    unittest.main()