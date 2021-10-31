import torch
import numpy as np
import time
import copy

class MLPModel(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs, activation=torch.nn.functional.leaky_relu, seed=None, pruning=None):
        super(MLPModel, self).__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.activation = activation
        self.seed = seed if seed is not None else int(time.time())
        self.pruning = pruning

        torch.random.manual_seed(self.seed)

        self.input_to_hidden = torch.nn.Linear(self.inputs, self.hidden)
        self.hidden_to_hidden = torch.nn.Linear(self.hidden, self.hidden)
        self.hidden_to_out = torch.nn.Linear(self.hidden, self.outputs)

        self.pruned = False
        self.initial_state = copy.deepcopy(self.state_dict())

    def get_trainable_parameters_number(self):
        ret = 0
        for parameter in self.parameters():
            ret += (parameter != 0).float().sum().item()
        return ret

    def forward(self, X):
        h1 = self.activation(self.input_to_hidden(X))
        h2 = self.activation(self.hidden_to_hidden(h1))
        y = self.activation(self.hidden_to_out(h2))

        return y

    def compute_prune_mask(self, p):
        toprune = int(p.nelement()*self.pruning)
        mask = torch.ones(*p.shape)
        topk = torch.topk(torch.abs(p).view(-1), k=toprune, largest=False)
        mask.view(-1)[topk.indices] = 0
        
        return mask

    def compute_prune_masks(self):
        self.input_to_hidden_weight_mask = self.compute_prune_mask(self.input_to_hidden.weight)
        self.input_to_hidden_bias_mask = self.compute_prune_mask(self.input_to_hidden.bias)
        self.hidden_to_hidden_weight_mask = self.compute_prune_mask(self.hidden_to_hidden.weight)
        self.hidden_to_hidden_bias_mask = self.compute_prune_mask(self.hidden_to_hidden.bias)
        self.hidden_to_out_weight_mask = self.compute_prune_mask(self.hidden_to_out.weight)
        self.hidden_to_out_bias_mask = self.compute_prune_mask(self.hidden_to_out.bias)
    
    def prune(self, compute_masks=True):
        if compute_masks: self.compute_prune_masks()

        self.input_to_hidden.weight.data = self.input_to_hidden.weight.data*self.input_to_hidden_weight_mask
        self.input_to_hidden.bias.data = self.input_to_hidden.bias.data*self.input_to_hidden_bias_mask
        self.hidden_to_hidden.weight.data = self.hidden_to_hidden.weight.data*self.hidden_to_hidden_weight_mask
        self.hidden_to_hidden.bias.data = self.hidden_to_hidden.bias.data*self.hidden_to_hidden_bias_mask
        self.hidden_to_out.weight.data = self.hidden_to_out.weight.data*self.hidden_to_out_weight_mask
        self.hidden_to_out.bias.data = self.hidden_to_out.bias.data*self.hidden_to_out_bias_mask

        self.pruned = True
    
    def reinit(self):
        self.load_state_dict(self.initial_state)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward(*args, **kwargs)
        if self.pruned:

            self.input_to_hidden.weight.grad = self.input_to_hidden.weight.grad*self.input_to_hidden_weight_mask
            self.input_to_hidden.bias.grad = self.input_to_hidden.bias.grad*self.input_to_hidden_bias_mask
            self.hidden_to_hidden.weight.grad = self.hidden_to_hidden.weight.grad*self.hidden_to_hidden_weight_mask
            self.hidden_to_hidden.bias.grad = self.hidden_to_hidden.bias.grad*self.hidden_to_hidden_bias_mask
            self.hidden_to_out.weight.grad = self.hidden_to_out.weight.grad*self.hidden_to_out_weight_mask
            self.hidden_to_out.bias.grad = self.hidden_to_out.bias.grad*self.hidden_to_out_bias_mask
