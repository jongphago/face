# Configuration
from arcface_torch.configs.aihub_r50_onegpu import config as aihub_config
from arcface_torch.configs.base import config as cfg
from nia_age.main_ae import LAMBDA_1, LAMBDA_2, START_AGE, END_AGE, NUM_AGE_GROUPS
from easydict import EasyDict as edict


def update_config(config, wandb_logger):
    config.lr = wandb_logger.config.lr
    config.num_epoch = wandb_logger.config.num_epoch
    config.weight.face = wandb_logger.config.face_weight
    config.weight.age = wandb_logger.config.age_weight
    config.weight.age_group = wandb_logger.config.age_group_weight
    config.weight.age_mean_var = wandb_logger.config.age_mean_var_weight
    return config


cfg.update(aihub_config)
cfg.output = "work_dirs/aihub_r50_onegpu"
cfg.num_classes = 2154

cfg.momentum = 0.9  #
cfg.weight_decay = 5e-4  #
cfg.lr = 0.02

cfg.is_fr, cfg.is_ae, cfg.is_kr = [
    True,
    True,
    False,
]

cfg.total_step = 2900
cfg.warmup_step = 0

cfg.num_train_family = 700
cfg.num_ages = END_AGE - START_AGE + 1
cfg.num_age_groups = NUM_AGE_GROUPS

cfg.scale = 64

cfg.lambdas = LAMBDA_1, LAMBDA_2
cfg.start_age = START_AGE
cfg.end_age = END_AGE

cfg.weight = edict(
    {
        "age": 1.0,
        "face": 1.0,
        "age_group": 1.0,
        "age_mean_var": 1.0,
        "kinship": 1.0,
    }
)

cfg.log_interval = 10
cfg.num_epoch = 2
