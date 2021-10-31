import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class CombinedJointUNNEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer, pl_module):
        logs = trainer.logger_connector.callback_metrics
        num_networks = trainer.model.model.networks

        current = logs.get(self.monitor)
        trainer.dev_debugger.track_early_stopping_history(self, current)

        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current, device=pl_module.device)

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience and num_networks == 1
            if num_networks > 1:
                trainer.model.model.selection()
                self.wait_count = 0

            if bool(should_stop):
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        should_stop = trainer.accelerator_backend.early_stopping_should_stop(pl_module)
        trainer.should_stop = should_stop