import torch
import numpy as np

import pytorch_lightning as pl
from sklearn.metrics import classification_report

from src.models.abstractclassifier import AbstractClassifier
from src.models.mnn.model import MNNModel

class MNNClassifierModel(AbstractClassifier):
    def __init__(self, config):
        super(MNNClassifierModel, self).__init__()
        self.save_hyperparameters(config)

        self.model = MNNModel(
            inputs=config.model.hyperparameters.inputs,
            hidden=config.model.hyperparameters.hidden,
            outputs=config.model.hyperparameters.outputs,
            pruning=config.model.hyperparameters.pruning,
            ticks=config.model.hyperparameters.ticks,
            activation=config.model.hyperparameters.activation
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        outputs = self.model.forward(X)
        outputs = torch.softmax(outputs, dim=1)
        
        return outputs

    def training_step(self, batch, batch_nb):
        X, y = batch
        out = self(X)

        loss = self.loss_fn(out, y)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y).float().mean()

        self.log("train/loss", loss)
        self.log("train/accuracy", acc)

        return loss

    def backward(self, *args, **kwargs):
        if hasattr(self.model, "backward"):
            return self.model.backward(*args, **kwargs)
        else:
            return super(MNNClassifierModel, self).backward(*args, **kwargs)