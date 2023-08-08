import torch
import argparse
import numpy as np
import os
import json

from sklearn import datasets as skdatasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models.cjunn.classifier import CombinedJointUNNClassifierModel
from src.models.cjunn.earlystop import CombinedJointUNNEarlyStopping
from src.models.mlp2.classifier import MLP2ClassifierModel
from src.models.mlp3.classifier import MLP3ClassifierModel
from src.models.mlp4.classifier import MLP4ClassifierModel


from src.utils import Namespace
from configs import datasets, models


def get_datasets(config):
    X, y = skdatasets.fetch_openml(config.dataset.name, return_X_y=True, as_frame=False, version=config.dataset.version if hasattr(config.dataset, "version") else "active")
    classes = np.unique(y).tolist()
    y = np.array([classes.index(i) for i in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-config.dataset.train_perc), random_state=config.dataset.seed)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=config.dataset.validation_perc, random_state=config.dataset.seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train))
    validation_dataset = torch.utils.data.TensorDataset(torch.tensor(X_validation).float(), torch.tensor(y_validation))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.dataset.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.dataset.batch_size)

    return Namespace(dict(
        train=train_dataloader,
        validation=validation_dataloader,
        test=test_dataloader
    ))


def get_model(config, args):
    if config.model.name == "cjunn":
        return CombinedJointUNNClassifierModel(
            config,
            plot_network=args.plot_network,
            plot_network_each=args.plot_network_each,
            plot_network_tmp=args.plot_network_tmp,
            compute_stats=args.compute_stats,
            compute_stats_each=args.compute_stats_each
        )
    elif config.model.name == "mlp2":
        return MLP2ClassifierModel(config)
    elif config.model.name == "mlp3":
        return MLP3ClassifierModel(config)    
    elif config.model.name == "mlp4":
        return MLP4ClassifierModel(config)
    else:
        raise Exception("Unknown model '%s'" % (config.model.name))


def get_callbacks(config, args):
    callbacks = []
    if config.model.name == "cjunn":
        early_stop_callback = CombinedJointUNNEarlyStopping(
            monitor="validation/epoch_loss",
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min"
        )
        callbacks.append(early_stop_callback)
    elif config.model.name == "mlp2" or config.model.name == "mlp3" or config.model.name == "mlp4":
        early_stop_callback = EarlyStopping(
            monitor="validation/epoch_loss",
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min"
        )
        callbacks.append(early_stop_callback)
    else:
        raise Exception("Unknown model '%s'" % (config.model.name))

    return callbacks



def run_train(config, args, datasets, loggers):    
    config.model.hyperparameters.inputs = config.dataset.features
    config.model.hyperparameters.outputs = config.dataset.classes
    config.model.hyperparameters.pruning = args.pruning
    
    model = get_model(config, args)
    callbacks = get_callbacks(config, args)
    
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=config.dataset.max_epochs, logger=loggers, callbacks=callbacks)
    
    trainer.fit(model, datasets.train, datasets.validation)
    
    if args.prune or args.lottery_ticket:
        prune_mask = model.prune()
    
    if args.lottery_ticket:
        callbacks = get_callbacks(config, args)
        
        model.reinit()
        model.prune(prune_mask)

        trainer = pl.Trainer(gpus=args.gpus, max_epochs=config.dataset.max_epochs, logger=loggers, callbacks=callbacks)
        trainer.fit(model, datasets.train, datasets.validation)
    
    if args.compute_stats:
        last_stats = model.visualizer.compute_stats()
    
    trainer.test(model, datasets.train)
    train_report = trainer.model.test_report
    
    trainer.test(model, datasets.validation)
    validation_report = trainer.model.test_report

    trainer.test(model, datasets.test)
    test_report = trainer.model.test_report

    return dict(
        model = model, 
        reports = dict(
            train = train_report,
            validation = validation_report,
            test = test_report
        ),
        stats=last_stats if args.compute_stats else None
    )

def train(config, args, replicas=None):
    datasets = get_datasets(config)
    loggers = []
    if args.log_tensorboard:
        loggers.append(pl.loggers.TensorBoardLogger("tb_logs", name="%s-%s" % (config.model.name, config.dataset.name)))
    if args.log_wandb:
        loggers.append(pl.loggers.WandbLogger(project=args.wandb_project, entity=args.wandb_entity))

    if replicas is None:
        train_results = run_train(config, args, datasets, loggers)
        reports = train_results["reports"]
        
        reports["parameters"] = train_results["model"].model.get_trainable_parameters_number()
        reports["stats"] = train_results["stats"]
        print(json.dumps(reports, indent=4))
    else:
        results = dict(
            train = dict(
                accuracy=np.zeros(replicas).tolist(),
            ),
            validation = dict(
                accuracy=np.zeros(replicas).tolist(),
            ),
            test = dict(
                accuracy=np.zeros(replicas).tolist(),
            ),
            topology = dict(
                parameters=np.zeros(replicas).tolist(),
            ),
            stats=dict(
                mean_path_length=np.zeros(replicas).tolist(),
                std_path_length=np.zeros(replicas).tolist(),
                min_path_length=np.zeros(replicas).tolist(),
                max_path_length=np.zeros(replicas).tolist(),
            )
        )
        all_reports = list()
        for i in range(0, replicas):
            train_results = run_train(config, args, datasets, loggers)
            
            results["train"]["accuracy"][i] = train_results["reports"]["train"]["accuracy"]
            results["validation"]["accuracy"][i] = train_results["reports"]["validation"]["accuracy"]
            results["test"]["accuracy"][i] = train_results["reports"]["test"]["accuracy"]
            results["topology"]["parameters"][i] = train_results["model"].model.get_trainable_parameters_number()

            if args.compute_stats:
                results["stats"]["mean_path_length"][i] = train_results["stats"]["mean_path_length"]
                results["stats"]["std_path_length"][i] = train_results["stats"]["std_path_length"]
                results["stats"]["min_path_length"][i] = train_results["stats"]["min_path_length"]
                results["stats"]["max_path_length"][i] = train_results["stats"]["max_path_length"]

            all_reports.append(train_results["reports"])
            
        results_stats = dict()
        for group in results:
            for metric in results[group]:
                results_stats["results/%s/%s/mean" % (group, metric)] = np.mean(results[group][metric])
                results_stats["results/%s/%s/std" % (group, metric)] = np.std(results[group][metric])
        
        print("=== RESULTS STATS ===")
        print(json.dumps(results_stats, indent=4))
        print("=== RESULTS ===")
        print(json.dumps(results, indent=4))
        print("=== REPORTS ===")
        print(json.dumps(all_reports, indent=4))
        print(" === END ===")

        if args.wandb_log_results:
            import wandb    
            reports_filename = os.path.join("/tmp", "%s_reports.json" % (wandb.run.id))
            reports_file = open(reports_filename, "w")
            json.dump(all_reports, reports_file)
            reports_file.close()

            results_filename = os.path.join("/tmp", "%s_results.json" % (wandb.run.id))
            results_file = open(results_filename, "w")
            json.dump(results, results_file)
            results_file.close()

            wandb.log(results_stats)

            wandb.save(reports_filename)
            wandb.save(results_filename)
        
        return results_stats
        

def get_default_args():
    return argparse.Namespace(
        gpus=0,
        patience=10,
        plot_network=False,
        plot_network_each=50,
        plot_network_tmp="/tmp/plot-network-tmp",
        compute_stats=False,
        compute_stats_each=50,
        log_tensorboard=False,
        log_wandb=False,
        pruning=0,
        prune=False,
        lottery_ticket=False,
        wandb_project="cjunn",
        wandb_entity="mlpi",
        wandb_log_results=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    defaults = get_default_args()

    parser.add_argument("--model", type=str, choices=list(models.keys()), required=True)
    parser.add_argument("--dataset", type=str, choices=list(datasets.keys()), required=True)
    
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--pruning", type=float, default=0)
    parser.add_argument("--lottery-ticket", action="store_true")

    parser.add_argument("--gpus", type=int, default=defaults.gpus)
    parser.add_argument("--patience", type=int, default=defaults.patience)

    parser.add_argument("--plot-network", action="store_true", default=defaults.plot_network)
    parser.add_argument("--plot-network-each", type=int, default=defaults.plot_network_each)
    parser.add_argument("--plot-network-tmp", type=str, default=defaults.plot_network_tmp)

    parser.add_argument("--compute-stats", action="store_true", default=defaults.compute_stats)
    parser.add_argument("--compute-stats-each", type=int, default=defaults.compute_stats_each)
    
    parser.add_argument("--log-tensorboard", action="store_true", default=defaults.log_tensorboard)
    parser.add_argument("--log-wandb", action="store_true", default=defaults.log_wandb)

    parser.add_argument("--wandb-project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb-entity", type=str, default=defaults.wandb_entity)
    parser.add_argument("--wandb-log-results", action="store_true", default=defaults.wandb_log_results)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.plot_network_tmp):
        os.mkdir(args.plot_network_tmp)

    config = Namespace(dict(
        model=models[args.model],
        dataset=datasets[args.dataset]
    ))
    train(config, args)