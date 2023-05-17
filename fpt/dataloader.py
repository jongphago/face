from fpt.dataset import (
    face_age_train_dataset,
    face_age_valid_dataset,
    face_age_test_dataset,
    aihub_pairs_valid_dataset,
    aihub_pairs_test_dataset,
)
from torch.utils.data import DataLoader

train_batch_size = 32
valid_batch_size = 1
test_batch_size = 1

face_age_train_loader = DataLoader(
    face_age_train_dataset,
    batch_size=train_batch_size,
    num_workers=0,
    shuffle=True,
)

face_age_valid_loader = DataLoader(
    face_age_valid_dataset,
    batch_size=valid_batch_size,
    num_workers=0,
    shuffle=False,
)

face_age_test_loader = DataLoader(
    face_age_test_dataset,
    batch_size=test_batch_size,
    num_workers=0,
    shuffle=False,
)

aihub_pairs_valid_loader = DataLoader(
    aihub_pairs_valid_dataset,
    batch_size=valid_batch_size,
    num_workers=0,
    shuffle=False,
)

aihub_pairs_test_loader = DataLoader(
    aihub_pairs_test_dataset,
    batch_size=test_batch_size,
    num_workers=0,
    shuffle=False,
)
