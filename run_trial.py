import optuna
import copy
import argparse
import os

from src.utils import Namespace
from configs import datasets, models
from train import train, get_default_args

def sample_params(trial):
    patience = trial.suggest_int("patience", 5, 50)
    if config["model"]["name"] == "cjunn":
        hidden = trial.suggest_int("hidden", 1, 10)
        max_incoming_connections = trial.suggest_int("max_incoming_connections", 1, 20)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
        mutation_entropy = trial.suggest_loguniform("mutation_entropy", 1e-4, 1e-1)

        return dict(
            args=dict(
                patience=patience,
            ),
            config=dict(
                hidden=hidden,
                max_incoming_connections=max_incoming_connections,
                lr=lr,
                mutation_entropy=mutation_entropy
            )
        )
    elif config["model"]["name"] == "mlp":
        hidden  = trial.suggest_int("hidden", 1, 50)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)

        return dict(
            args=dict(
                patience=patience
            ),
            config=dict(
                hidden=hidden,
                lr=lr,
            )
        )
  
    else:
        raise Exception("Unknown model: '%s'" % (config.model.name, ))



def objective(trial):
    hyperparameters = sample_params(trial)
    
    trial_config = copy.deepcopy(config)
    trial_config["model"]["hyperparameters"].update(hyperparameters["config"])
    trial_config = Namespace(trial_config)
    trial_config.study_name = args.study_name
    trial_config.opt_id = args.opt_id
    trial_config.patience = hyperparameters["args"]["patience"]
    trial_config.replicas = args.optuna_replicas
    

    trial_args = get_default_args()
    trial_args.patience = hyperparameters["args"]["patience"]
    trial_args.log_wandb = args.wandb
    trial_args.wandb_log_results = args.wandb
    trial_args.pruning = args.pruning
    trial_args.prune = args.prune
    trial_args.lottery_ticket = args.lottery_ticket

    results = train(trial_config, trial_args, args.optuna_replicas)
    return results["results/validation/accuracy/mean"]


parser = argparse.ArgumentParser()

parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--wandb", action="store_true")

args = parser.parse_args()

study = optuna.load_study(study_name=args.study_name, storage=os.environ["OPTUNA_STORAGE"])
vars(args).update(study.user_attrs["args"])

config = dict(
    dataset=datasets[args.dataset],
    model=models[args.model]
)

study.optimize(objective, n_trials=1)