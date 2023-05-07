import torch
import numpy as np
import time
import copy

class MLP3Model(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs, activation=torch.nn.functional.leaky_relu, seed=None, pruning=None):
        super(MLP3Model, self).__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.activation = activation
        self.seed = seed if seed is not None else int(time.time())
        self.pruning = pruning

        torch.random.manual_seed(self.seed)

        self.input_to_hidden1 = torch.nn.Linear(self.inputs, self.hidden)
        self.hidden1_to_hidden2 = torch.nn.Linear(self.hidden, self.hidden)
        self.hidden2_to_hidden3 = torch.nn.Linear(self.hidden, self.hidden)
        self.hidden3_to_out = torch.nn.Linear(self.hidden, self.outputs)
        
        self.pruned = False
        self.initial_state = copy.deepcopy(self.state_dict())

    def get_trainable_parameters_number(self):
        ret = 0
        for parameter in self.parameters():
            ret += (parameter != 0).float().sum().item()
        return ret

    def forward(self, X):
        h1 = self.activation(self.input_to_hidden1(X))
        h2 = self.activation(self.hidden1_to_hidden2(h1))
        h3 = self.activation(self.hidden2_to_hidden3(h2))
        y = self.activation(self.hidden3_to_out(h3))

        return y

    def compute_prune_mask(self, p):
        toprune = int(p.nelement()*self.pruning)
        mask = torch.ones(*p.shape)
        topk = torch.topk(torch.abs(p).view(-1), k=toprune, largest=False)
        mask.view(-1)[topk.indices] = 0
        
        return mask

    def compute_prune_masks(self):
        self.input_to_hidden1_weight_mask = self.compute_prune_mask(self.input_to_hidden1.weight)
        self.input_to_hidden1_bias_mask = self.compute_prune_mask(self.input_to_hidden1.bias)

        self.hidden1_to_hidden2_weight_mask = self.compute_prune_mask(self.hidden1_to_hidden2.weight)
        self.hidden1_to_hidden2_bias_mask = self.compute_prune_mask(self.hidden1_to_hidden2.bias)

        self.hidden2_to_hidden3_weight_mask = self.compute_prune_mask(self.hidden2_to_hidden3.weight)
        self.hidden2_to_hidden3_bias_mask = self.compute_prune_mask(self.hidden2_to_hidden3.bias)

        self.hidden3_to_out_weight_mask = self.compute_prune_mask(self.hidden3_to_out.weight)
        self.hidden3_to_out_bias_mask = self.compute_prune_mask(self.hidden3_to_out.bias)
    
    def prune(self, compute_masks=True):
        if compute_masks: self.compute_prune_masks()

        self.input_to_hidden1.weight.data = self.input_to_hidden1.weight.data*self.input_to_hidden1_weight_mask
        self.input_to_hidden1.bias.data = self.input_to_hidden1.bias.data*self.input_to_hidden1_bias_mask

        self.hidden1_to_hidden2.weight.data = self.hidden1_to_hidden2.weight.data*self.hidden1_to_hidden2_weight_mask
        self.hidden1_to_hidden2.bias.data = self.hidden1_to_hidden2.bias.data*self.hidden1_to_hidden2_bias_mask

        self.hidden2_to_hidden3.weight.data = self.hidden2_to_hidden3.weight.data*self.hidden2_to_hidden3_weight_mask
        self.hidden2_to_hidden3.bias.data = self.hidden2_to_hidden3.bias.data*self.hidden2_to_hidden3_bias_mask

        self.hidden3_to_out.weight.data = self.hidden4_to_out.weight.data*self.hidden3_to_out_weight_mask
        self.hidden3_to_out.bias.data = self.hidden4_to_out.bias.data*self.hidden3_to_out_bias_mask

        self.pruned = True
    
    def reinit(self):
        self.load_state_dict(self.initial_state)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward(*args, **kwargs)
        if self.pruned:
            self.input_to_hidden1.weight.grad = self.input_to_hidden1.weight.grad*self.input_to_hidden1_weight_mask
            self.input_to_hidden1.bias.grad = self.input_to_hidden1.bias.grad*self.input_to_hidden1_bias_mask

            self.hidden1_to_hidden2.weight.grad = self.hidden1_to_hidden2.weight.grad*self.hidden1_to_hidden2_weight_mask
            self.hidden1_to_hidden2.bias.grad = self.hidden1_to_hidden2.bias.grad*self.hidden1_to_hidden2_bias_mask

            self.hidden2_to_hidden3.weight.grad = self.hidden2_to_hidden3.weight.grad*self.hidden2_to_hidden3_weight_mask
            self.hidden2_to_hidden3.bias.grad = self.hidden2_to_hidden3.bias.grad*self.hidden2_to_hidden3_bias_mask

            self.hidden3_to_out.weight.grad = self.hidden3_to_out.weight.grad*self.hidden3_to_out_weight_mask
            self.hidden3_to_out.bias.grad = self.hidden3_to_out.bias.grad*self.hidden3_to_out_bias_mask