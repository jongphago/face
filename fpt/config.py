# Configuration
from arcface_torch.configs.aihub_r50_onegpu import config as aihub_config
from arcface_torch.configs.base import config as cfg

cfg.update(aihub_config)
cfg.output = "work_dirs/aihub_r50_onegpu"
cfg.num_classes = 2154

cfg.momentum = 0.9  #
cfg.weight_decay = 5e-4  #
cfg.lr = 0.02

cfg.is_fr = True
cfg.is_ae = True
cfg.is_kr = True

cfg.total_step = 1452
cfg.warmup_step = 0
