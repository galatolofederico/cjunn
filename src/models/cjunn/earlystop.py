import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class CombinedJointUNNEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer):
        logs = trainer.callback_metrics
        num_networks = trainer.model.model.networks
        
        if trainer.fast_dev_run or not self._validate_condition_metric(logs):
            return

        current = logs.get(self.monitor)

        trainer.dev_debugger.track_early_stopping_history(self, current)

        should_stop, reason = self._evaluate_stopping_criteria(current)
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop

        if should_stop and num_networks > 1:
            trainer.model.model.selection()
            self.wait_count = 0
            should_stop = False

        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)


        

