# Configuration
from arcface_torch.configs.aihub_r50_onegpu import config as aihub_config
from arcface_torch.configs.base import config as cfg
from nia_age.main_ae import LAMBDA_1, LAMBDA_2, START_AGE, END_AGE, NUM_AGE_GROUPS
from easydict import EasyDict as edict


def update_config(config, wandb_logger):
    config.lr = wandb_logger.config.lr
    config.num_epoch = wandb_logger.config.num_epoch
    config.margin_list = wandb_logger.config.margin_list
    config.network = wandb_logger.config.network
    config.embedding_size = wandb_logger.config.embedding_size
    config.sample_rate = wandb_logger.config.sample_rate
    config.momentum = wandb_logger.config.momentum
    config.weight_decay = wandb_logger.config.weight_decay
    config.dropout = wandb_logger.config.dropout
    config.optimizer = wandb_logger.config.optimizer
    return config


cfg.update(aihub_config)
cfg.project_name = ""
cfg.is_debug = False

cfg.output = "work_dirs/aihub_r50_onegpu"
cfg.num_classes = 2154

cfg.momentum = 0.9  #
cfg.weight_decay = 5e-4  #
cfg.lr = 0.02

cfg.is_fr, cfg.is_ae, cfg.is_kr = [
    None,
    None,
    None,
]

cfg.num_losses = 0
if cfg.is_fr:
    cfg.num_losses += 1
if cfg.is_ae:
    cfg.num_losses += 3
if cfg.is_kr:
    cfg.num_losses += 1

cfg.total_step = 2900
cfg.warmup_step = 0

cfg.num_train_family = 700
cfg.num_ages = END_AGE - START_AGE + 1
cfg.num_age_groups = NUM_AGE_GROUPS

cfg.scale = 64

cfg.lambdas = LAMBDA_1, LAMBDA_2
cfg.start_age = START_AGE
cfg.end_age = END_AGE

cfg.log_interval = 10
cfg.num_epoch = 2
