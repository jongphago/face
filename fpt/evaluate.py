import torch
from fpt.utils import tensor_to_int
from fpt.config import cfg


def evaluate(
    dataloader,
    model,
):
    for key, value in model.items():
        model[key] = value.eval()

    for index, sample in enumerate(dataloader):
        embeddings = model.embedding(sample.image.cuda())
        if cfg.is_fr:
            fr_pred = model.face(embeddings)
        if cfg.is_ae:
            age_pred, age_group_pred = model.age(embeddings)
        if cfg.is_kr:
            kinship_pred = model.kinship(embeddings)
        pass

        break


