import os
from typing import Callable
import numpy as np
from easydict import EasyDict as edict
import torch
from torch import distributed
from torch.nn import CrossEntropyLoss
from arcface_torch.losses import CombinedMarginLoss
from arcface_torch.partial_fc_v2 import PartialFC_V2
from nia_age.mean_variance_loss import MeanVarianceLoss
from nia_age.main_ae import LAMBDA_1, LAMBDA_2, START_AGE, END_AGE
from fpt.config import cfg


def face_loss_func(face_pred, sample, loss):
    # labels
    labels = sample.face_label.cuda()
    labels.squeeze_()
    labels = labels.long()
    labels = labels.view(-1, 1)

    # softmax
    softmax = loss.margin_loss(face_pred, labels)

    return loss.cross_entropy_loss(softmax, labels.flatten())


def age_loss_func(age_pred, age_group_pred, sample, mean_variance_loss, criterion):
    dta = np.array(sample.data_type)
    age_sample_indices = dta != "Age"
    age_pred = age_pred[age_sample_indices]
    labels = sample.age[age_sample_indices].cuda()

    mean_loss, variance_loss = mean_variance_loss(age_pred, labels)
    age_softmax_loss = criterion(age_pred, labels)
    mean_loss, variance_loss, age_softmax_loss

    age_group_pred = age_group_pred[~age_sample_indices]
    age_group_labels = sample.age_class[~age_sample_indices].cuda()
    age_group_softmax_loss = criterion(age_group_pred, age_group_labels)
    return age_softmax_loss, age_group_softmax_loss


def kinship_loss_func(kinship_pred, sample, criterion):
    labels = sample.family_class.cuda()
    kinship_softmax_loss = criterion(kinship_pred, labels)
    return kinship_softmax_loss


margin_loss = CombinedMarginLoss(
    64,
    *cfg.margin_list,
    cfg.interclass_filtering_threshold,
)


mean_variance_loss = MeanVarianceLoss(
    LAMBDA_1,
    LAMBDA_2,
    START_AGE,
    END_AGE,
)

cross_entropy_loss = CrossEntropyLoss()

if not distributed.is_initialized():
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")

    except KeyError:
        rank = 0
        local_rank = 0
        world_size = 1
        distributed.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )


module_partial_fc = PartialFC_V2(
    margin_loss,
    cfg.embedding_size,
    cfg.num_classes,
    cfg.sample_rate,
    cfg.fp16,
)

loss = edict(
    {
        "face": face_loss_func,
        "age": age_loss_func,
        "kinship": kinship_loss_func,
        "margin_loss": margin_loss.cuda(),
        "cross_entropy_loss": cross_entropy_loss.cuda(),
    }
)
