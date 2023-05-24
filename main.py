from pprint import pprint
from torch.optim import SGD
import numpy as np
from fpt.loss import Loss
from fpt.model import Model
from fpt.train import train
from fpt.config import cfg
from fpt.global_step import GlobalStep
from fpt.logger import initialize_wandb
from fpt.utils import log_verification_output, add_parameters
from fpt.dataloader import train_loader, pairs_valid_loader, pairs_test_loader
from facenet.validate_aihub import validate_aihub
from arcface_torch.lr_scheduler import PolyScheduler


if __name__ == "__main__":
    print(cfg)
    num_train_steps = len(train_loader)
    wandb_logger = initialize_wandb(cfg)
    loss = Loss(cfg)
    model = Model(cfg)
    optimizer = SGD(
        params=add_parameters(cfg, model, loss),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = PolyScheduler(
        optimizer=optimizer,
        base_lr=cfg.lr,
        max_steps=num_train_steps * cfg.num_epoch,  # 2900 * num_epoch
        warmup_steps=cfg.warmup_step,
        last_epoch=-1,
    )

    global_step = GlobalStep()
    for epoch in range(1, cfg.num_epoch + 1):
        wandb_logger.log({"Epoch": epoch})
        print(f"Current epoch: {epoch}/{cfg.num_epoch}")
        train(
            train_loader,
            loss,
            model,
            optimizer,
            lr_scheduler,
            cfg,
            wandb_logger,
            epoch,
            global_step,
        )
        validation_output = validate_aihub(
            model.embedding, pairs_valid_loader, cfg.network, epoch, task="Valid/"
        )
        log_verification_output(
            validation_output, wandb_logger, "Valid", global_step.get()
        )
    test_output = validate_aihub(
        model.embedding, pairs_test_loader, cfg.network, epoch, task="Test/"
    )
    log_verification_output(test_output, wandb_logger, "Test", global_step.get())
