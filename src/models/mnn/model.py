import torch
import numpy as np
import time

class MNNModel(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs, pruning=0, ticks=2, activation=torch.nn.functional.leaky_relu, seed=None):
        super(MNNModel, self).__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.pruning = pruning
        self.ticks = ticks
        self.activation = activation
        self.seed = seed if seed is not None else int(time.time())

        torch.random.manual_seed(self.seed)

        self.A = torch.nn.Linear(self.inputs + self.hidden, self.hidden + self.outputs)
        
        self.A_mask = torch.rand(self.inputs + self.hidden, self.hidden + self.outputs)
        self.A_mask = self.A_mask <= self.pruning

        self.A.weight.data[self.A_mask.T] = 0


    def get_trainable_parameters_number(self):
        ret = 0
        for parameter in self.parameters():
            ret += (parameter != 0).float().sum().item()
        return ret

    def forward(self, X):
        state = torch.zeros(X.shape[0], self.hidden).to(X.device)

        for t in range(0, self.ticks):
            input = torch.cat((state[:, :self.hidden], X), dim=1)
            state = self.activation(self.A(input))

        return torch.softmax(state[:,self.hidden:], dim=1)
    
    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward(*args, **kwargs)
        self.A.weight.grad[self.A_mask.T] = 0