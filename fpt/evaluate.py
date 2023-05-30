import argparse
from torch.utils.data import DataLoader
from fpt.path import DATA
from fpt.config import cfg
from fpt.model import Model
from fpt.dataset import AIHubDataset
from fpt.logger import initialize_wandb
from fpt.utils import log_verification_output
from fpt.transform import aihub_valid_transforms
from facenet.validate_aihub import validate_aihub


def evaluate(task, config, checkpoint, project_name=None):
    # config
    if project_name is not None:
        config.project_name = project_name
    
    # dataloader
    pairs_dataset = AIHubDataset(
        dir=DATA / "face-image/test_aihub_family",
        pairs_path=DATA / f"pairs/test/pairs_{task.upper()}.txt",
        transform=aihub_valid_transforms,
    )
    case1c_test_loader = DataLoader(pairs_dataset, batch_size=config.pairs_batch_size)

    # logger
    wandb_logger = initialize_wandb(config)

    # model
    model = Model(config)
    model_path = f"/home/jongphago/family-photo-tree/work_dirs/aihub_r50_onegpu/{checkpoint}_ArcFace/model.pt"
    model.load_embedding(path=model_path)

    # evaluate
    validate_output = validate_aihub(
        model.embedding, case1c_test_loader, "r50", 1, task=task
    )

    # logging
    log_verification_output(
        validate_output, wandb_logger, task.capitalize(), 0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--project_name', required=False)
    args = parser.parse_args()

    cfg.project_name = args.project_name if args.project_name else "log_test_validation"
    evaluate(args.task, cfg, args.checkpoint, cfg.project_name)
    