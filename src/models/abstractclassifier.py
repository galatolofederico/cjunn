import torch
import numpy as np

import pytorch_lightning as pl
from sklearn.metrics import classification_report

class AbstractClassifier(pl.LightningModule):
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        predictions = torch.argmax(logits, dim=1)

        return dict(
            rights = y.detach().cpu().numpy(),
            predictions = predictions.detach().cpu().numpy(),
        )

    def test_epoch_end(self, outputs):
        rights = np.concatenate([output["rights"] for output in outputs])
        predictions = np.concatenate([output["predictions"] for output in outputs])
        self.test_report = classification_report(rights, predictions, output_dict=True, zero_division=0)
        return self.test_report
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        predictions = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        self.log("validation/loss", loss.item())
        
        return dict(
            loss = loss.item(),
            rights = y.detach().cpu().numpy(),
            predictions = predictions.detach().cpu().numpy(),
        )
    
    def validation_epoch_end(self, outputs):
        report = self.test_epoch_end(outputs)
        losses = [output["loss"] for output in outputs]

        self.log("validation/epoch_loss", np.array(losses).mean())
        self.log("validation/accuracy", report["accuracy"])
        return report
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.model.hyperparameters.lr)
