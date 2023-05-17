# Configuration
from arcface_torch.configs.aihub_r50_onegpu import config as aihub_config
from arcface_torch.configs.base import config as cfg

cfg.update(aihub_config)
cfg.output = "work_dirs/aihub_r50_onegpu"
cfg.num_classes = 2154