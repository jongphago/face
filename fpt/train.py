import torch
from fpt.utils import tensor_to_int
from fpt.config import cfg
from fpt.path import DTFR
from fpt.data import join_face_df
from fpt.loss import mean_variance_loss, cross_entropy_loss


face_df = join_face_df(DTFR, "aihub_family")


def train(
    dataloader,
    losses,
    model,
    optimizer,
    lr_scheduler,
):
    for index, sample in enumerate(dataloader):
        embeddings = model.embedding(sample.image.cuda())
        loss = 0
        if cfg.is_fr:
            fr_loss: torch.Tensor = losses.face(embeddings, sample.face_label.cuda())
            loss += fr_loss
        if cfg.is_ae:
            age_pred, age_group_pred = model.age(embeddings)
            age_loss, age_group_loss = losses.age(
                age_pred,
                age_group_pred,
                sample,
                mean_variance_loss,
                cross_entropy_loss,
            )
            loss += age_loss
            loss += age_group_loss
        if cfg.is_kr:
            kinship_pred = model.kinship(embeddings)
            kinship_loss = losses.kinship(kinship_pred, sample, cross_entropy_loss)
            loss += kinship_loss

        if index % 10 == 0:
            print(f"{index:4d},", end=" ")
            if cfg.is_fr:
                print(f"fr: {tensor_to_int(fr_loss):4.2f}", end=" ")
            if cfg.is_ae:
                print(
                    f"age: {tensor_to_int(age_loss):4.2f}, age_group: {tensor_to_int(age_group_loss):4.2f}",
                    end=" ",
                )
            if cfg.is_kr:
                print(f"kinship: {tensor_to_int(kinship_loss):4.2f}", end=" ")
            print("")

        torch.nn.utils.clip_grad_norm_(model.embedding.parameters(), 5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # break
