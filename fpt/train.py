import torch
from fpt.utils import tensor_to_int
from fpt.config import cfg
from fpt.path import DTFR
from fpt.data import join_face_df
from fpt.model import face_age_model, age_module, kinship_module
from fpt.loss import age_loss_func, kinship_loss_func
from fpt.loss import module_partial_fc, mean_variance_loss, cross_entropy_loss

is_fr = True
is_ae = True
is_kr = True

face_df = join_face_df(DTFR, "aihub_family")

face_age_model.train().cuda()
age_module.train().cuda()
kinship_module.train().cuda()
module_partial_fc.train().cuda()


def train(face_age_train_loader, face_age_optimizer, lr_scheduler):
    for _, sample in enumerate(face_age_train_loader):
        embeddings = face_age_model(sample.image.cuda())
        loss = 0
        if is_fr:
            age_pred, age_group_pred = age_module(embeddings)
            fr_loss: torch.Tensor = module_partial_fc(
                embeddings, sample.face_label.cuda()
            )
            loss += fr_loss
        if is_ae:
            age_loss, age_group_loss = age_loss_func(
                age_pred, age_group_pred, sample, mean_variance_loss, cross_entropy_loss
            )
            loss += age_loss
            loss += age_group_loss
        if is_kr:
            kinship_pred = kinship_module(embeddings)
            kinship_loss = kinship_loss_func(kinship_pred, sample, cross_entropy_loss)
            loss += kinship_loss

        if _ % 10 == 0:
            print(
                f"{_:4d},\
                loss: {tensor_to_int(loss):8.4f},\
                fr: {tensor_to_int(fr_loss):4.2f},\
                age: {tensor_to_int(age_loss):4.2f},\
                age_group: {tensor_to_int(age_group_loss):4.2f}\
                kinship: {tensor_to_int(kinship_loss):4.2f}"
            )

        torch.nn.utils.clip_grad_norm_(face_age_model.parameters(), 5)
        face_age_optimizer.zero_grad()
        loss.backward()
        face_age_optimizer.step()
        lr_scheduler.step()
        break
