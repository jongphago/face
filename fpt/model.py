import torch
from torch import nn
from torch.nn.functional import normalize, linear
from arcface_torch.backbones import get_model


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


class Model:
    def __init__(self, config):
        self.config = config
        self.face = None
        self.embedding = None
        self.age = None
        self.kinship = None
        self.load()

    def get_embedding(self, dropout=0.0):
        self.embedding = get_model(self.config.network, dropout=dropout)

    def load_embedding(self, path=None):
        if path is None:
            path = f"/home/jupyter/family-photo-tree/utils/model/arcface/{self.config.network}/backbone.pth"
        self.embedding.load_state_dict(torch.load(path))

    def get_face_module(self):
        self.face = FaceRecogFC(self.config.embedding_size, self.config.num_classes)

    def get_age_module(self):
        self.age = AgeModule(self.config.num_ages, self.config.num_age_groups)

    def load_age_module(self, path=None):
        if path is None:
            path = "/home/jongphago/nia_age/result_model/model_0"
        saved_params = torch.load(path)
        selected_params = {
            k: v
            for k, v in saved_params.items()
            if "age_classifier" in k or "age_group_classifier" in k
        }
        self.age.load_state_dict(selected_params)

    def get_kinship_module(self):
        self.kinship = KinshipModule(self.config.num_train_family)

    def load(self):
        self.get_embedding()
        self.load_embedding()
        self.embedding.cuda()
        if self.config.is_fr:
            self.get_face_module()
        if self.config.is_ae:
            self.get_age_module()
            self.load_age_module()
            self.age.cuda()
        if self.config.is_kr:
            self.get_kinship_module()
            self.kinship.cuda()

    def to_train(self):
        self.embedding.train()
        if self.config.is_fr:
            self.face.train()
        if self.config.is_ae:
            self.age.train()
        if self.config.is_kr:
            self.kinship.train()

    def to_eval(self):
        self.embedding.eval()
        if self.config.is_fr:
            self.face.eval()
        if self.config.is_ae:
            self.age.eval()
        if self.config.is_fr:
            self.kinship.train()
