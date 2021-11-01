import torch

datasets = dict(
    iris = dict(
        name = "iris",
        features = 4,
        classes = 3,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    credit = dict(
        name = "credit-g",
        features = 20,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    transfusion = dict(
        name = "blood-transfusion-service-center",
        features = 4,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    monks = dict(
        name = "monks-problems-1",
        features = 6,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    tictactoe = dict(
        name = "tic-tac-toe",
        features = 9,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    plates = dict(
        name = "steel-plates-fault",
        features = 33,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    krkp = dict(
        name = "kr-vs-kp",
        features = 36,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    qsar = dict(
        name = "qsar-biodeg",
        features = 41,
        classes = 2,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    nursery = dict(
        name = "nursery",
        version = 3,
        features = 8,
        classes = 4,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    robot = dict(
        name = "wall-robot-navigation",
        features = 24,
        classes = 4,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
    seeds = dict(
        name = "seeds",
        features = 7,
        classes = 3,
        batch_size = 16,
        train_perc = 0.8,
        validation_perc = 0.25,
        max_epochs = 300,
        seed = 42,
    ),
)


models = dict(
    sb = dict(
        name = "self-boosted",
        hyperparameters = dict(
            hidden = 5,
            max_incoming_connections = 10,
            activation = torch.nn.functional.leaky_relu,
            activation_th = 0,
            networks = 10,
            lr = .01,
            mutation_entropy = 1e-1,
            start_mutating = 10,
        )
    ),
    mnn = dict(
        name = "mnn",
        hyperparameters = dict(
            hidden = 10,
            pruning = 0.5,
            ticks = 2,
            activation = torch.nn.functional.leaky_relu,
            lr = .01
        )
    ),
    mlp = dict(
        name = "mlp",
        hyperparameters = dict(
            hidden = 10,
            activation = torch.nn.functional.leaky_relu,
            lr = .01,
        )
    )
)


configs = dict(
    cjunn_iris = dict(
        dataset = datasets["iris"],
        model = models["sb"]
    ),
    cjunn_credit=dict(
        dataset = datasets["credit"],
        model = models["sb"]
    ),
    cjunn_transfusion=dict(
        dataset = datasets["transfusion"],
        model = models["sb"]
    ),
    cjunn_monks=dict(
        dataset = datasets["monks"],
        model = models["sb"]
    ),
    cjunn_tictactoe=dict(
        dataset = datasets["tictactoe"],
        model = models["sb"]
    ),
    cjunn_plates=dict(
        dataset = datasets["plates"],
        model = models["sb"]
    ),
    cjunn_krkp=dict(
        dataset = datasets["krkp"],
        model = models["sb"]
    ),
    cjunn_qsar=dict(
        dataset = datasets["qsar"],
        model = models["sb"]
    ),
    cjunn_nursery=dict(
        dataset = datasets["nursery"],
        model = models["sb"]
    ),
    cjunn_robot=dict(
        dataset = datasets["robot"],
        model = models["sb"]
    ),
    cjunn_seeds=dict(
        dataset = datasets["seeds"],
        model = models["sb"]
    ),

    mnn_iris = dict(
        dataset = datasets["iris"],
        model = models["mnn"]
    ),
    mnn_credit=dict(
        dataset = datasets["credit"],
        model = models["mnn"]
    ),
    mnn_transfusion=dict(
        dataset = datasets["transfusion"],
        model = models["mnn"]
    ),
    mnn_monks=dict(
        dataset = datasets["monks"],
        model = models["mnn"]
    ),
    mnn_tictactoe=dict(
        dataset = datasets["tictactoe"],
        model = models["mnn"]
    ),
    mnn_plates=dict(
        dataset = datasets["plates"],
        model = models["mnn"]
    ),
    mnn_krkp=dict(
        dataset = datasets["krkp"],
        model = models["mnn"]
    ),
    mnn_qsar=dict(
        dataset = datasets["qsar"],
        model = models["mnn"]
    ),
    mnn_nursery=dict(
        dataset = datasets["nursery"],
        model = models["mnn"]
    ),
    mnn_robot=dict(
        dataset = datasets["robot"],
        model = models["mnn"]
    ),
    mnn_seeds=dict(
        dataset = datasets["seeds"],
        model = models["mnn"]
    ),

    mlp_iris = dict(
        dataset = datasets["iris"],
        model = models["mlp"]
    ),
    mlp_credit=dict(
        dataset = datasets["credit"],
        model = models["mlp"]
    ),
    mlp_transfusion=dict(
        dataset = datasets["transfusion"],
        model = models["mlp"]
    ),
    mlp_monks=dict(
        dataset = datasets["monks"],
        model = models["mlp"]
    ),
    mlp_tictactoe=dict(
        dataset = datasets["tictactoe"],
        model = models["mlp"]
    ),
    mlp_plates=dict(
        dataset = datasets["plates"],
        model = models["mlp"]
    ),
    mlp_krkp=dict(
        dataset = datasets["krkp"],
        model = models["mlp"]
    ),
    mlp_qsar=dict(
        dataset = datasets["qsar"],
        model = models["mlp"]
    ),
    mlp_nursery=dict(
        dataset = datasets["nursery"],
        model = models["mlp"]
    ),
    mlp_robot=dict(
        dataset = datasets["robot"],
        model = models["mlp"]
    ),
    mlp_seeds=dict(
        dataset = datasets["seeds"],
        model = models["mlp"]
    ),
)



__all__ = dict(
    datasets=datasets,
    models=models,
    configs=configs
)