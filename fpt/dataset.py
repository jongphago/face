import os
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from facenet.datasets.AIHubDataset import AIHubDataset
from nia_age.data import NiaDataset as nia
from fpt.transform import nia_train_transforms, nia_valid_transforms
from fpt.transform import (
    aihub_train_transforms,
    aihub_valid_transforms,
    aihub_test_transforms,
)
from fpt.path import DTFR
from fpt.data import join_face_df

face_df = join_face_df(DTFR, "aihub_family")


RANGE_TO_MEDIAN = nia.RANGE_TO_MEDIAN
AGE_GROUPS = nia.AGE_GROUPS
GROUP_TO_INDEX = {group: index for index, group in enumerate(AGE_GROUPS)}
TRAIN_FID_TO_INDEX = {idx: f"F{i:04d}" for idx, i in enumerate(range(1, 701))}
age_to_age_groups = nia.age_to_age_groups


class FaceAgeDataset(Dataset):
    def __init__(self, root_dir, face_df, transform):
        super(FaceAgeDataset, self).__init__()
        self.face_dataset = ImageFolder(root=root_dir, transform=transform)
        self.face_df = face_df
        self.class_to_idx = self.face_dataset.class_to_idx
        self.samples = self.face_dataset.samples
        uuids = [
            img_path.rsplit(".", 1)[0].rsplit("/", 1)[1] for img_path, _ in self.samples
        ]
        unique_family_id = self.face_df.loc[uuids].family_id.unique()
        self.FID_TO_INDEX = {
            id: index for index, id in enumerate(sorted(unique_family_id))
        }

    def __len__(self):
        return len(self.face_dataset)

    def __getitem__(self, index):
        image, face_label = self.face_dataset[index]
        path, _ = self.face_dataset.samples[index]
        *_, key = os.path.splitext(path)[0].split("/")
        row = self.face_df.loc[key]
        sample = edict(
            {
                "image": image,
                "age": row.age,
                "age_class": GROUP_TO_INDEX[row.age_group],
                "file": path,
                "data_type": row.category,
                "family_id": row.family_id,
                "family_class": self.FID_TO_INDEX[row.family_id],
                "personal_id": row.target,
                "face_label": face_label,
                "key": key,
            }
        )
        return sample


face_age_train_dataset = FaceAgeDataset(
    root_dir="/home/jupyter/data/face-image/train_aihub_family",
    face_df=face_df,
    transform=aihub_train_transforms,
)

face_age_valid_dataset = FaceAgeDataset(
    root_dir="/home/jupyter/data/face-image/valid_aihub_family",
    face_df=face_df,
    transform=aihub_valid_transforms,
)

face_age_test_dataset = FaceAgeDataset(
    root_dir="/home/jupyter/data/face-image/test_aihub_family",
    face_df=face_df,
    transform=aihub_test_transforms,
)

aihub_pairs_valid_dataset = AIHubDataset(
    dir="/home/jupyter/data/face-image/valid_aihub_family",
    pairs_path="/home/jupyter/data/pairs/valid/pairs_Age.txt",
    transform=aihub_valid_transforms,
)

aihub_pairs_test_dataset = AIHubDataset(
    dir="/home/jupyter/data/face-image/test_aihub_family",
    pairs_path="/home/jupyter/data/pairs/test/pairs_Age.txt",
    transform=aihub_valid_transforms,
)

aihub_pairs_sample_dataset = AIHubDataset(
    dir="/home/jupyter/data/face-image/test_aihub_family",
    pairs_path="/home/jupyter/data/pairs/sample/pairs_Age.txt",
    transform=aihub_valid_transforms,
)
