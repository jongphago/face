import os
from pathlib import Path
from uuid import uuid4
from fpt.utils import get_folder_name
import pandas as pd

ROOT = Path("/home/jongphago/family-photo-tree")
DATA = ROOT / "data"
FACE = DATA / "face-image"
DICT = DATA / "dict"


def get_face_target_path(key, c, category):
    family_id = key[:5]
    folder_name = get_folder_name(family_id)
    target = (
        FACE
        / category
        / folder_name
        / family_id
        / f"{family_id}-{c}"
        / f"{str(uuid4())}.jpg"
    )
    dirname = os.path.dirname(target)
    os.makedirs(dirname, exist_ok=True)
    return target


def get_face_image_path_from_series(series: pd.Series) -> Path:
    return (
        FACE
        / series.data_category
        / series.folder_name
        / series.family_id
        / f"{series.family_id}-{series.personal_id}"
        / f"{series.name}.jpg"
    )
