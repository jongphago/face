from easydict import EasyDict as edict
from fpt.config import cfg
from fpt.path import DTFR
from fpt.data import join_face_df
from fpt.model import model
from fpt.loss import loss
from fpt.dataloader import face_age_train_loader as train_loader
from fpt.dataloader import face_age_valid_loader as valid_loader
from fpt.dataloader import aihub_pairs_valid_loader as pair_valid_loader
from fpt.train import train
from fpt.evaluate import evaluate, evaluate_fv
from torch.optim import SGD
from arcface_torch.lr_scheduler import PolyScheduler


face_df = join_face_df(DTFR, "aihub_family")


if __name__ == "__main__":
    optimizer = SGD(
        params=[
            {"params": model.embedding.parameters()},
            {"params": model.face.parameters()},
            {"params": model.kinship.parameters()},
        ],
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    lr_scheduler = PolyScheduler(
        optimizer=optimizer,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,  # 1452
        warmup_steps=cfg.warmup_step,
        last_epoch=-1,
    )

    for key, value in model.items():
        model[key] = value.cuda()

    for current_epoch in range(1, cfg.num_epoch + 1):
        print(f"Current epoch: {current_epoch}/{cfg.num_epoch}")

        train(
            train_loader,
            loss,
            model,
            optimizer,
            lr_scheduler,
        )

        evaluate(
            valid_loader,
            model,
        )
        
        evaluate_fv(
            pair_valid_loader,
            model,
        )
