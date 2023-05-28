from fpt.config import cfg

sweep_configuration = {
    "name": cfg.project_name,
    "method": "bayes",
    "metric": {"name": "Valid/accuracy", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 1},
    "parameters": {},
}

default_params = {
    "lr": {"values": [0.2, 0.02, 0.002]},
    "margin_list": {
        "values": [
            (1.0, 0.5, 0.0),
        ]
    },
    "network": {"values": ["r50"]},
    "embedding_size": {"values": [512]},
    "sample_rate": {"min": 0.7, "max": 1.0, "distribution": "uniform"},
    "momentum": {
        "values": [
            0.9,
            0.99,
            0.999,
        ]
    },
    "weight_decay": {
        "values": [
            5e-5,
            5e-4,
            5e-3,
        ]
    },
    "dropout": {"min": 0.0, "max": 0.5, "distribution": "uniform"},
    "num_epoch": {"values": [5, 10]},
    "optimizer": {
        "values": [
            "sgd",
        ]
    },
}

fr_params = {}
age_params = {}
kr_params = {}

params = sweep_configuration["parameters"]
params.update(default_params)

if cfg.is_fr:
    params.update(fr_params)
if cfg.is_fr:
    params.update(fr_params)
if cfg.is_fr:
    params.update(fr_params)
