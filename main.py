from pprint import pprint
from torch.optim import SGD
import numpy as np
import wandb
from fpt.loss import Loss
from fpt.model import Model
from fpt.train import train
from fpt.config import cfg, update_config
from fpt.global_step import GlobalStep
from fpt.logger import initialize_wandb
from fpt.sweep import sweep_configuration
from fpt.utils import log_verification_output, add_parameters
from fpt.save import save_checkpoint
from fpt.dataloader import (
    train_loader,
    pairs_valid_loader,
    pairs_test_loader,
    pairs_sample_loader,
    case1_test_loader,
)
from facenet.validate_aihub import validate_aihub
from arcface_torch.lr_scheduler import PolyScheduler


def main(cfg):
    wandb_logger = initialize_wandb(cfg)
    if not cfg.is_debug:
        cfg = update_config(cfg, wandb_logger)
    num_train_steps = len(train_loader)
    pprint(cfg)
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
    best_valid_score = -1
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
        valid_accuracy = np.mean(validation_output[-1][0])
        if valid_accuracy > best_valid_score:
            best_valid_score = valid_accuracy
            save_checkpoint(
                loss,
                model,
                optimizer,
                lr_scheduler,
                cfg,
                wandb_logger,
                epoch,
                global_step,
            )
    test_output = validate_aihub(
        model.embedding, pairs_test_loader, cfg.network, epoch, task="Test/"
    )
    log_verification_output(test_output, wandb_logger, "Test", global_step.get())
    case1_output = validate_aihub(
        model.embedding, case1_test_loader, cfg.network, epoch, task="case1/"
    )
    log_verification_output(case1_output, wandb_logger, "Case1", global_step.get())


if __name__ == "__main__":
    if cfg.is_debug:
        main(cfg)
    else:
        sweep_id = wandb.sweep(
            sweep_configuration,
            entity="jongphago",
            project=cfg.project_name,
        )
        wandb.agent(sweep_id, lambda: main(cfg))
