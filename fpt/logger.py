import wandb
from datetime import datetime


def initialize_wandb(config):
    wandb_logger = None
    if config.using_wandb:
        # Sign in to wandb
        try:
            wandb.login(key=config.wandb_key, relogin=True)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")

        # Initialize wandb
        if config.run_name is not None:
            run_name = config.run_name
        else:
            run_name = datetime.now().strftime("%y%m%d_%H%M")
            run_name = (
                run_name
                if config.suffix_run_name is None
                else run_name + f"_{config.suffix_run_name}"
            )
        try:
            wandb_logger = (
                wandb.init(
                    entity=config.wandb_entity,
                    project=config.project_name,
                    sync_tensorboard=True,
                    resume=config.wandb_resume,
                    name=run_name,
                    config=config,
                )
                if config.wandb_log_all
                else None
            )
        except Exception as e:
            print(
                "WandB Data (Entity and Project name) must be provided in config file (base.py)."
            )
            print(f"Config Error: {e}")

        if wandb_logger:
            pass  # your custom code here if required

    return wandb_logger
