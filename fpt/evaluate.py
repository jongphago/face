import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import torch
import wandb
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.nn.modules.distance import PairwiseDistance
from fpt.path import DATA
from fpt.config import cfg
from fpt.model import Model
from fpt.dataset import AIHubDataset
from fpt.logger import initialize_wandb
from fpt.transform import aihub_valid_transforms


def get_checkpoint_info(model_path):
    dirname = os.path.dirname(model_path)
    json_path = os.path.join(dirname, "best.json")
    with open(json_path, "r") as f:
        checkpoint_dict = json.load(f)
        checkpoint_dict = edict(checkpoint_dict)
    return checkpoint_dict


def evaluate(tasks, config, checkpoint, project_name=None):
    # config
    now = datetime.now().strftime("%y%m%d_%H%M")
    if project_name is not None:
        config.project_name = project_name
    pairs_batch_size = 32
    config.wandb_resume = False

    # distance metric
    l2_distance = PairwiseDistance(p=2)

    # model
    model = Model(config)
    model_path = f"/home/jongphago/family-photo-tree/work_dirs/aihub_r50_onegpu/{checkpoint}_ArcFace/model.pt"
    model.load_embedding(path=model_path)

    # checkpoint dict
    ckp_dict = get_checkpoint_info(model_path)
    best_distance = ckp_dict.best_distance
    config.run_name = f"{ckp_dict.about}-{ckp_dict.checkpoint}-{now}"
    config["about"] = ckp_dict.about
    config["best_distance"] = ckp_dict.best_distance

    # logger
    logger = initialize_wandb(config)

    tasks = [tasks] if isinstance(tasks, str) else tasks
    acc_dict = defaultdict(int)
    for task in tasks:
        # dataloader
        pairs_dataset = AIHubDataset(
            dir=DATA / "face-image/test_aihub_family",
            pairs_path=DATA / f"pairs/test/pairs_{task.upper()}.txt",
            transform=aihub_valid_transforms,
        )
        test_loader = DataLoader(pairs_dataset, batch_size=pairs_batch_size)

        # evaluate
        out = 0
        for a, b, label in tqdm(test_loader):
            output_a = model.embedding(a.cuda())
            output_b = model.embedding(b.cuda())
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance
            result = torch.eq(distance.cpu().detach() < best_distance, label)
            out += result.sum().detach()
        

        # logging
        accuracy = out / len(test_loader.dataset)
        print(f"{task.capitalize()}/accuracy: {accuracy:4.2%}")
        acc_dict[f"{task.capitalize()}/accuracy"] = accuracy
        
    logger.log(acc_dict)
    logger.finish()


if __name__ == "__main__":
    if False:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tasks", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--project_name", required=False)
        args = parser.parse_args()

        cfg.project_name = (
            args.project_name if args.project_name else "log_test_validation"
        )
        evaluate(args.tasks, cfg, args.checkpoint, cfg.project_name)
    else:
        tasks = [
            "BASIC-G",
            "BASIC-GC",
            "BASIC-A",
            "BASIC-AC",
            "BASIC-F",
            "BASIC-FN",
            "BASIC-FC",
            "FAMILY-A",
            "FAMILY-CA",
            "FAMILY-G",
            "FAMILY-CG",
            "FAMILY-AG",
            "FAMILY-CAG",
            "PERSONAL-A",
            "PERSONAL-AC",
        ]
        evaluate(tasks, cfg, "230529_0140", "test_validation")  # single
        evaluate(tasks, cfg, "230602_0236", "test_validation")  # dual
        evaluate(tasks, cfg, "230601_1838", "test_validation")  # triple
