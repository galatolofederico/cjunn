import torch
import numpy as np
import os

import pytorch_lightning as pl
from sklearn.metrics import classification_report

from src.models.abstractclassifier import AbstractClassifier
from src.models.cjunn.model import CombinedJointUNNModel
from src.visualizer import Visualizer

class CombinedJointUNNClassifierModel(AbstractClassifier):
    def __init__(self, config, plot_network=False, plot_network_each=None, plot_network_tmp=None, compute_stats=False, compute_stats_each=None):
        super(CombinedJointUNNClassifierModel, self).__init__()
        self.save_hyperparameters(config)
        
        self.plot_network = plot_network
        self.plot_network_each = plot_network_each
        self.plot_network_tmp = plot_network_tmp

        self.compute_stats = compute_stats
        self.compute_stats_each = compute_stats_each

        self.model = CombinedJointUNNModel(
            inputs=config.model.hyperparameters.inputs,
            hidden=config.model.hyperparameters.hidden,
            outputs=config.model.hyperparameters.outputs,
            networks=config.model.hyperparameters.networks,
            max_incoming_connections=config.model.hyperparameters.max_incoming_connections,
            activation=config.model.hyperparameters.activation,
            mutation_entropy=config.model.hyperparameters.mutation_entropy,
            activation_th=config.model.hyperparameters.activation_th
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if self.plot_network or self.compute_stats:
            self.visualizer = Visualizer(self.model)
            self.last_stats = dict(
                mean_path_length=0,
                std_path_length=0,
                min_path_length=0,
                max_path_length=0,
            )

        self.training_step_number = 0
        self.mutation_runs = 0

    def forward(self, X):
        outputs = self.model.forward(X, weighted=False)
        outputs = torch.softmax(outputs, dim=2)
        
        networks_weights = self.model.get_networks_weights().unsqueeze(0).unsqueeze(2)
        outputs = (networks_weights * outputs).sum(dim=1)
        return outputs


    def training_step(self, batch, batch_nb):
        X, y = batch
        out = self(X)

        loss = self.loss_fn(out, y)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y).float().mean()


        if self.current_epoch > self.hparams.model.hyperparameters.start_mutating:
            if self.model.mutation():
                self.mutation_runs += 1
        

        if self.plot_network and self.global_step % self.plot_network_each == 0:
            self.visualizer.save(os.path.join(self.plot_network_tmp, "network_%d.png" % (self.global_step, )))

        self.log("train/loss", loss)
        self.log("train/accuracy", acc, prog_bar=True)
        
        self.log("misc/networks", self.model.connections.shape[0], prog_bar=True)
        self.log("misc/max_weight", self.model.get_networks_weights().max())
        self.log("misc/min_weight", self.model.get_networks_weights().min())
        self.log("misc/mutation_runs", self.mutation_runs, prog_bar=True)

        if self.compute_stats and self.global_step % self.compute_stats_each == 0:
            self.last_stats = self.visualizer.compute_stats()

        self.log("stats/mean_path_length", self.last_stats["mean_path_length"], prog_bar=True)
        self.log("stats/std_path_length", self.last_stats["std_path_length"])
        self.log("stats/max_path_length", self.last_stats["max_path_length"])
        self.log("stats/min_path_length", self.last_stats["min_path_length"])

        return loss
