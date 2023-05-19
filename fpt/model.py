from re import X
from easydict import EasyDict as edict
import torch
from torch import nn
from torch.nn.functional import normalize, linear
from arcface_torch.backbones import get_model
from fpt.config import cfg
from nia_age.main_ae import AgeModel
from nia_age.main_ae import START_AGE, END_AGE, NUM_AGE_GROUPS

NUM_TRAIN_FAMILY = 700
NUM_AGES = END_AGE - START_AGE + 1
network = "r50"
NUM_CLASSES = cfg.num_classes


class FaceRecogFC(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
    ):
        super(FaceRecogFC, self).__init__()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (1, embedding_size)))
        self.num_classes = num_classes

    def forward(self, x):
        # embeddings
        norm_embeddings = normalize(x)
        # weight
        weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_classes, 512))
        ).cuda()
        norm_weight_activated = normalize(weight)
        norm_weight_activated.shape
        # logits
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)
        return logits


class AgeModule(nn.Module):
    def __init__(self, num_ages, num_age_groups):
        super(AgeModule, self).__init__()
        self.age_classifier = nn.Linear(512, num_ages)
        self.age_group_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_age_groups),
        )

    def forward(self, x):
        age_pred = self.age_classifier(x)
        age_group_pred = self.age_group_classifier(x)
        return age_pred, age_group_pred


class KinshipModule(nn.Module):
    def __init__(self, num_family_id):
        super(KinshipModule, self).__init__()
        self.kinship_classifier = nn.Linear(512, num_family_id)

    def forward(self, x):
        kinship_pred = self.kinship_classifier(x)
        return kinship_pred


face_age_model = get_model(network, dropout=0.0)
face_age_path = (
    f"/home/jupyter/family-photo-tree/utils/model/arcface/{network}/backbone.pth"
)
face_age_model.load_state_dict(torch.load(face_age_path))

face_module = FaceRecogFC(512, NUM_CLASSES)
nia_age_path = "/home/jongphago/nia_age/result_model/model_0"
age_module = AgeModule(NUM_AGES, NUM_AGE_GROUPS)
saved_params = torch.load(nia_age_path)
selected_params = {
    k: v
    for k, v in saved_params.items()
    if "age_classifier" in k or "age_group_classifier" in k
}
age_module.load_state_dict(selected_params)
kinship_module = KinshipModule(NUM_TRAIN_FAMILY)

model = edict(
    {
        "embedding": face_age_model.cuda(),
        "face": face_module.cuda(),
        "age": age_module.cuda(),
        "kinship": kinship_module.cuda(),
    }
)
