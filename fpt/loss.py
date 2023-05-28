import os
import numpy as np
import torch
from torch import distributed
from torch.nn import CrossEntropyLoss
from arcface_torch.losses import CombinedMarginLoss
from arcface_torch.partial_fc_v2 import PartialFC_V2
from nia_age.mean_variance_loss import MeanVarianceLoss

from torch import nn


class MultiLossLayer(nn.Module):
    def __init__(self, list_size):
        super(MultiLossLayer, self).__init__()
        self.list_size = list_size
        self.log_vars = nn.Parameter(torch.zeros(list_size))
        self.loss_lis = None

    def forward(self, losses):
        dtype = self.log_vars.dtype
        self.loss_list = [
            torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
            for i in range(self.list_size)
        ]
        return sum(self.loss_list)


class Loss:
    def __init__(self, config):
        self.config = config
        self.margin_loss = None
        self.multi_loss_layer = None
        self.module_partial_fc = None
        self.cross_entropy_loss = CrossEntropyLoss()
        self.define()

    def define_margin_loss(self):
        self.margin_loss = CombinedMarginLoss(
            self.config.scale,
            *self.config.margin_list,
            self.config.interclass_filtering_threshold,
        )

    def define_mean_variance_loss(self):
        self.mean_variance_loss = MeanVarianceLoss(
            *self.config.lambdas,
            self.config.start_age,
            self.config.end_age,
        )

    def define_partial_fc(self):
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

        self.module_partial_fc = PartialFC_V2(
            self.margin_loss,
            self.config.embedding_size,
            self.config.num_classes,
            self.config.sample_rate,
            self.config.fp16,
        ).cuda()

    def face(self, pred, sample):
        # labels
        labels = sample.face_label.cuda()
        labels.squeeze_()
        labels = labels.long()
        labels = labels.view(-1, 1)
        # softmax
        softmax = self.margin_loss(pred, labels)
        return self.cross_entropy_loss(softmax, labels.flatten())

    def age(self, pred, sample):
        age_pred, age_group_pred = pred
        data_type = np.array(sample.data_type)
        age_sample_indices = data_type != "Age"

        age_pred = age_pred[age_sample_indices]
        age_labels = sample.age[age_sample_indices].cuda()
        age_loss = self.cross_entropy_loss(age_pred, age_labels)

        mean_variance_loss = self.mean_variance_loss(age_pred, age_labels)
        weighted_mean_variance_loss = sum(
            a * b for a, b in zip(self.config.lambdas, mean_variance_loss)
        )

        age_group_pred = age_group_pred[~age_sample_indices]
        age_group_labels = sample.age_class[~age_sample_indices].cuda()
        age_group_loss = self.cross_entropy_loss(age_group_pred, age_group_labels)
        return age_loss, age_group_loss, weighted_mean_variance_loss

    def kinship(self, pred, sample):
        labels = sample.family_class.cuda()
        return self.cross_entropy_loss(pred, labels)

    def define(self):
        self.define_margin_loss()
        self.define_mean_variance_loss()
        self.define_partial_fc()
        self.multi_loss_layer = MultiLossLayer(list_size=self.config.num_losses)  # number of losses
