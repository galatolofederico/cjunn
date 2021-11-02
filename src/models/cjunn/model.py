import torch
import numpy as np
import time

class CombinedJointUNNModel(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs, networks, max_incoming_connections, activation=torch.nn.functional.leaky_relu, seed=None, mutation_entropy=1e-1, activation_th=0):
        super(CombinedJointUNNModel, self).__init__()
        self.zero_bias = inputs
        self.one_bias = inputs + 1
        self.inputs = inputs + 2
        self.hidden = hidden
        self.outputs = outputs
        self.networks = networks
        self.max_incoming_connections = max_incoming_connections

        self.activation = activation
        self.activation_th = activation_th
        self.seed = seed if seed is not None else int(time.time())
        self.mutation_entropy = mutation_entropy

        torch.random.manual_seed(self.seed)
        
        self.W = torch.nn.Parameter(torch.randn((self.inputs + self.hidden, self.hidden + self.outputs)))
        self.networks_weights = torch.nn.ParameterList()
        for i in range(0, self.networks):
            self.networks_weights.append(torch.nn.Parameter(torch.rand(1)))
        self.removed_networks = []
        self.activation_stats = np.zeros((self.networks, self.hidden + self.outputs))
        self.activation_stats_alpha = 0.01
        
        self.init_connections()

    def init_connections(self):
        self.connections = torch.zeros((self.networks, self.hidden+self.outputs, self.max_incoming_connections), dtype=int)
        for n in range(0, self.networks):
            for i in range(0, self.hidden+self.outputs):
                conns = torch.randint(0, min(self.inputs + i - 1, self.inputs+self.hidden) , (self.max_incoming_connections, ))
                conns = torch.unique(conns)

                incomings = torch.ones((self.max_incoming_connections, ))*self.zero_bias
                incomings[:conns.shape[0]] = conns

                self.connections[n, i] = incomings

    def get_networks_weights(self, set_removed_to_inf=False, apply_softmax=True):
        if set_removed_to_inf:
            networks_weights = torch.cat([self.networks_weights[i] if i not in self.removed_networks else torch.tensor([float("inf")]).to(self.networks_weights[0].device) for i in range(0, len(self.networks_weights))])
        else:
            networks_weights = torch.cat([self.networks_weights[i] for i in range(0, len(self.networks_weights)) if i not in self.removed_networks])
        if apply_softmax:
            return torch.softmax(networks_weights, dim=0)
        else:
            return networks_weights
    
    def get_trainable_parameters_number(self):
        return (self.connections != self.zero_bias).float().sum().item()

    def get_activation_entropy(self):
        eps = 1e-5
        return -np.log(self.activation_stats + eps)*self.activation_stats - np.log((1-self.activation_stats) + eps)*(1-self.activation_stats)
        
    def forward(self, inputs, weighted=True, return_postsynaptic=False):
        device = inputs.device
        batch_size = inputs.shape[0]
        inputs = torch.cat((inputs, torch.zeros(batch_size, 1).to(device), torch.ones(batch_size, 1).to(device)), dim=1).float()
        postsynaptic = torch.cat((inputs, torch.zeros((batch_size, self.hidden+self.outputs)).to(device)), dim=1).float()
        neurons = self.inputs + self.hidden + self.outputs
        postsynaptic = postsynaptic.reshape(batch_size, 1, neurons)
        postsynaptic = postsynaptic.repeat(1, self.networks, 1)
        connections = self.connections.to(device).clone()

        for node in range(0, self.hidden+self.outputs):
            incoming_indices = connections[:, node]

            incoming_indices_batch = incoming_indices.unsqueeze(0)
            incoming_indices_batch = incoming_indices_batch.repeat(batch_size, 1, 1)
            incoming_values = torch.gather(postsynaptic.clone(), 2, incoming_indices_batch)
            incoming_weights = self.W[incoming_indices, node]

            outgoing_values = incoming_weights * incoming_values
            outgoing_values = outgoing_values.sum(dim=2)

            postsynaptic[:, :, self.inputs + node] = self.activation(outgoing_values)
        


        postsynaptic_activations = (postsynaptic[:, :, self.inputs:] > self.activation_th).float().detach()
        postsynaptic_activations = postsynaptic_activations.mean(dim=0).cpu().numpy()

        self.activation_stats = self.activation_stats_alpha*postsynaptic_activations + (1-self.activation_stats_alpha)*self.activation_stats
        
        if return_postsynaptic:
            return postsynaptic

        outputs = postsynaptic[:, :, -self.outputs:]

        if not weighted:
            return outputs
        
        networks_weights = self.get_networks_weights().unsqueeze(0).unsqueeze(2)
        outputs = (networks_weights * outputs).sum(dim=1)
        return outputs

    def selection(self):
        if self.networks == 1:
            return False
        networks_weights = self.get_networks_weights()

        networks_weights_all = self.get_networks_weights(set_removed_to_inf=True, apply_softmax=False)
        # softmax preseve order
        min_weight_all = torch.argmin(networks_weights_all).item()
        min_weight = torch.argmin(networks_weights).item()

        self.connections = torch.cat((self.connections[:min_weight,:,:], self.connections[min_weight+1:,:,:]))
        self.activation_stats = np.concatenate((self.activation_stats[:min_weight], self.activation_stats[min_weight+1:]))
        self.removed_networks.append(min_weight_all)

        self.networks -= 1
        return True

    def mutation(self):
        connections = self.connections.cpu().numpy()
        mutation_run = False
        activation_entropy = self.get_activation_entropy()
        if np.all(activation_entropy >= self.mutation_entropy):
            return False
        for n, node in zip(*np.where(activation_entropy < self.mutation_entropy)):
            incoming_connections = connections[n, node]
            max_available_input_node = self.inputs + node

            # Only hidden nodes can be added or removed
            entropies = activation_entropy[n]
            entropies_hidden = entropies[:-self.outputs]

            sorted_entropies = np.argsort(entropies_hidden)
            sorted_entropies = sorted_entropies + self.inputs
            
            removed = False
            for remove_candidate in sorted_entropies:
                remove_id = np.where(incoming_connections == remove_candidate)[0]
                if len(remove_id) > 0:
                    assert len(remove_id) == 1
                    self.connections[n, node, remove_id[0]] = torch.tensor(self.zero_bias)
                    removed = True
                    break
            zeros = np.where(incoming_connections == self.zero_bias)[0]
            inserted = False
            if len(zeros) > 0:
                zero = zeros[0]
                for add_candidate in sorted_entropies[::-1]:
                    if add_candidate not in incoming_connections and add_candidate < max_available_input_node:
                        self.connections[n, node, zero] = torch.tensor(add_candidate)
                        inserted = True
                        break
                        
        return removed or inserted