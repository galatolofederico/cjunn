import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, model):
        self.model = model
        self.G = nx.DiGraph()
        self.clear()
        
        self.pos = nx.random_layout(self.G, center=(0, 0))

        min_x = min([n[0] for k, n in self.pos.items()])
        min_y = min([n[1] for k, n in self.pos.items()])
        max_x = max([n[0] for k, n in self.pos.items()])
        max_y = max([n[1] for k, n in self.pos.items()])
        
        inputs = list(range(0, self.model.inputs))
        inputs.remove(self.model.zero_bias)
        line_dim = (max_y - min_y)
        x_offset = 0.5
        fixed_nodes = inputs
        for i, node in enumerate(fixed_nodes):
            p = (i+1)/len(fixed_nodes)
            self.pos[node] = (min_x-x_offset, min_y + p*line_dim)
        fixed_nodes = list(range(self.model.inputs+self.model.hidden, self.model.inputs+self.model.hidden+self.model.outputs))
        for i, node in enumerate(fixed_nodes):
            p = (i+1)/len(fixed_nodes)
            self.pos[node] = (max_x+x_offset, min_y + p*line_dim)    

    def clear(self):
        self.G.clear()
        plt.close()
        activation_entropy = self.model.get_activation_entropy().mean(axis=0)
        activation_entropy = (activation_entropy - activation_entropy.min()) / (activation_entropy.max() - activation_entropy.min())
        for i in range(0, self.model.inputs+self.model.hidden+self.model.outputs):
            if i < self.model.inputs:
                alpha = 1
            else:
                alpha = activation_entropy[i-self.model.inputs]
            if i != self.model.zero_bias:
                self.G.add_node(i, alpha=alpha)
        plt.figure()

    def save(self, file):
        W = self.model.W.detach().cpu().numpy()
        norm_weights = (W - W.min()) / (W.max() - W.min())
        network_weights = self.model.get_networks_weights().detach().cpu().numpy()
        networks_connections = self.model.connections.detach().cpu().numpy()
        
        self.clear()
        for connections, network_weight in zip(networks_connections, network_weights):
            for node in range(0, self.model.hidden+self.model.outputs):
                incoming_connections = connections[node]
                for f in incoming_connections:
                    if f != self.model.zero_bias:
                        self.G.add_edge(f, self.model.inputs+node, weight=norm_weights[f, node], alpha=(1 - network_weight) if self.model.networks > 1 else 1)
        
        nodes = self.G.nodes(data=True)
        edges = self.G.edges(data=True)

        weights = [e[2]["weight"] for e in edges]
        edges_alphas = [e[2]["alpha"] for e in edges]
        nodes_alphas = [n[1]["alpha"] for n in nodes]

        edge_color = [(0, 0, 0, a) for a in edges_alphas]
        node_color = [(0.3, 0.6, 0.9, a) for a in nodes_alphas]
        
        nx.draw(
            self.G,
            self.pos,
            width=weights,
            edge_color=edge_color,
            node_color=node_color,
            with_labels=True
        )

        plt.savefig(file)


    def compute_stats(self):
        W = self.model.W.detach().cpu().numpy()
        norm_weights = (W - W.min()) / (W.max() - W.min())
        network_weights = self.model.get_networks_weights().detach().cpu().numpy()
        networks_connections = self.model.connections.detach().cpu().numpy()
        
        self.clear()
        for connections, network_weight in zip(networks_connections, network_weights):
            for node in range(0, self.model.hidden+self.model.outputs):
                incoming_connections = connections[node]
                for f in incoming_connections:
                    if f != self.model.zero_bias:
                        self.G.add_edge(f, self.model.inputs+node, weight=norm_weights[f, node], alpha=(1 - network_weight) if self.model.networks > 1 else 1)
      
        print(self.G)