from torch.optim import SGD
from fpt.loss import Loss
from fpt.model import Model
from fpt.train import train
from fpt.config import cfg
from fpt.dataloader import train_loader, pairs_valid_loader, pairs_test_loader
from facenet.validate_aihub import validate_aihub
from arcface_torch.lr_scheduler import PolyScheduler


if __name__ == "__main__":
    loss = Loss(cfg)
    model = Model(cfg)
    optimizer = SGD(
        params=[
            {"params": model.age.parameters()},
            {"params": model.face.parameters()},
            {"params": model.kinship.parameters()},
            {"params": model.embedding.parameters()},
            {"params": loss.module_partial_fc.parameters()},
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

    for epoch in range(1, cfg.num_epoch + 1):
        print(f"Current epoch: {epoch}/{cfg.num_epoch}")
        train(train_loader, loss, model, optimizer, lr_scheduler, cfg)
        validate_aihub(model.embedding, pairs_valid_loader, cfg.network, epoch)
    validate_aihub(model.embedding, pairs_test_loader, cfg.network, epoch)
