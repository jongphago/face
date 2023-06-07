from torch.utils.data import DataLoader
from fpt.dataset import (
    face_age_train_dataset,
    face_age_valid_dataset,
    face_age_test_dataset,
    aihub_pairs_valid_dataset,
    aihub_pairs_test_dataset,
    aihub_pairs_sample_dataset,
    aihub_pairs_case1_dataset,
)

train_batch_size = 32
valid_batch_size = 1
test_batch_size = 1
pairs_batch_size = 512

train_loader = DataLoader(
    face_age_train_dataset,
    batch_size=train_batch_size,
    num_workers=0,
    shuffle=True,
    drop_last=True,
)

valid_loader = DataLoader(
    face_age_valid_dataset,
    batch_size=valid_batch_size,
    num_workers=0,
    shuffle=False,
)

test_loader = DataLoader(
    face_age_test_dataset,
    batch_size=test_batch_size,
    num_workers=0,
    shuffle=False,
)

pairs_valid_loader = DataLoader(
    aihub_pairs_valid_dataset,
    batch_size=pairs_batch_size,
    num_workers=0,
    shuffle=False,
)

pairs_test_loader = DataLoader(
    aihub_pairs_test_dataset,
    batch_size=pairs_batch_size,
    num_workers=0,
    shuffle=False,
)

pairs_sample_loader = DataLoader(
    aihub_pairs_sample_dataset,
    batch_size=pairs_batch_size,
    num_workers=0,
    shuffle=False,
)

case1_test_loader = DataLoader(
    aihub_pairs_case1_dataset,
    batch_size=pairs_batch_size,
    num_workers=0,
    shuffle=False,
)