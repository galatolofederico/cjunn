import optuna
import argparse
import os
from configs import models, datasets

parser = argparse.ArgumentParser()

parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--model", type=str, choices=list(models.keys()), required=True)
parser.add_argument("--dataset", type=str, choices=list(datasets.keys()), required=True)
parser.add_argument("--prune", action="store_true")
parser.add_argument("--pruning", type=float, default=0)
parser.add_argument("--lottery-ticket", action="store_true")

parser.add_argument("--opt-id", type=str, default="")

parser.add_argument("--optuna-sampler", type=str, default="TPESampler")
parser.add_argument("--optuna-pruner", type=str, default="SuccessiveHalvingPruner")
parser.add_argument("--optuna-replicas", type=int, default=10)


args = parser.parse_args()


study = optuna.create_study(
    study_name=args.study_name,
    direction="maximize",
    sampler=getattr(optuna.samplers, args.optuna_sampler)(),
    pruner=getattr(optuna.pruners, args.optuna_pruner)(),
    storage=os.environ["OPTUNA_STORAGE"]
)

study.set_user_attr("args", vars(args))
