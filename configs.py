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
    cjunn = dict(
        name = "cjunn",
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
    mlp2 = dict(
        name = "mlp2",
        hyperparameters = dict(
            hidden = 10,
            activation = torch.nn.functional.leaky_relu,
            lr = .01,
        )
    ),
    mlp3 = dict(
        name = "mlp3",
        hyperparameters = dict(
            hidden = 10,
            activation = torch.nn.functional.leaky_relu,
            lr = .01,
        )
    ),
    mlp4 = dict(
        name = "mlp4",
        hyperparameters = dict(
            hidden = 10,
            activation = torch.nn.functional.leaky_relu,
            lr = .01,
        )
    )
)


__all__ = dict(
    datasets=datasets,
    models=models,
)