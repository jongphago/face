import os
import torch
import wandb


def save_checkpoint(
    loss,
    model,
    optimizer,
    lr_scheduler,
    config,
    wandb_logger,
    epoch,
    global_step,
):
    output = os.path.join(config.output, wandb_logger.name)
    os.makedirs(output, exist_ok=True)
    if config.save_all_states:
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step.get(),
            "state_dict_backbone": model.embedding.state_dict(),
            "state_optimizer": optimizer.state_dict(),
            "state_lr_scheduler": lr_scheduler.state_dict(),
        }
        if config.is_fr:
            checkpoint.update(
                {
                    "state_dict_face": model.face.state_dict(),
                    "state_dict_softmax_fc": loss.module_partial_fc.state_dict(),
                }
            )
        if config.is_ae:
            checkpoint.update(
                {
                    "state_dict_age": model.age.state_dict(),
                }
            )
        if config.is_kr:
            checkpoint.update(
                {
                    "state_dict_kinship": model.kinship.state_dict(),
                }
            )
        torch.save(checkpoint, os.path.join(output, f"checkpoint_gpu.pt"))

    path_module = os.path.join(output, "model.pt")
    torch.save(model.embedding.state_dict(), path_module)

    if wandb_logger and config.save_artifacts:
        artifact_name = f"{wandb_logger.name}_E{epoch}"
        model = wandb.Artifact(artifact_name, type="model")
        model.add_file(path_module)
        wandb_logger.log_artifact(model)
