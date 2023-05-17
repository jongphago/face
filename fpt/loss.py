import os
from typing import Callable
import numpy as np
import torch
from torch import distributed
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize, linear
from arcface_torch.losses import CombinedMarginLoss
from arcface_torch.partial_fc_v2 import PartialFC_V2
from nia_age.mean_variance_loss import MeanVarianceLoss
from nia_age.main_ae import LAMBDA_1, LAMBDA_2, START_AGE, END_AGE
from fpt.config import cfg


NUM_CLASSES = cfg.num_classes


class FaceRecogFC(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
    ):
        super(FaceRecogFC, self).__init__()
        self.cross_entropy = CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (1, embedding_size)))
        self.margin_loss = margin_loss
        self.num_classes = num_classes

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        # labels
        labels.squeeze_()
        labels = labels.long()
        labels = labels.view(-1, 1)

        # embeddings
        norm_embeddings = normalize(embeddings)

        # weight
        weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_classes, 512))
        ).cuda()
        norm_weight_activated = normalize(weight)
        norm_weight_activated.shape

        # logits
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        # softmax
        softmax = self.margin_loss(logits, labels)

        # loss
        loss = self.cross_entropy(softmax, labels.flatten())

        return loss


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

face_recog_fc = FaceRecogFC(
    margin_loss,
    512,
    NUM_CLASSES,
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
