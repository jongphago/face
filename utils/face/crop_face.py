import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List
import numpy as np
from fpt.path import DICT, FACE, DTFR
from fpt.path import get_face_target_path
from fpt.data import crop_faces, init_df_sample_face, create_face_series, flatten_directory

# Crop family-sample face
FILE_DICT = "file_dict.txt"
BBOX_DICT = "bbox_dict.txt"
DATA_CATEGORY = "aihub_family"

# FILE_DICT = "sample_file_dict.txt"
# BBOX_DICT = "sample_bbox_dict.txt"
# DATA_CATEGORY = "aihub_sample"

## Data dictionary
### Open sample_file_dict
with open(DICT / FILE_DICT, "r") as f:
    sample_file_dict = json.load(f)

### Open sample_bbox_dict
with open(DICT / BBOX_DICT, "r") as f:
    sample_bbox_dict = json.load(f)


if __name__ == "__main__":
    df_sample_face = init_df_sample_face()
    for key, value in tqdm(sample_bbox_dict.items()):
        image_path = sample_file_dict[key]["image"]
        out = crop_faces(key, value, image_path)
        classes = [r[0] for r in value]
        for sliced_array, c in zip(out, classes):
            target = get_face_target_path(key, c, category=DATA_CATEGORY)
            face_image = Image.fromarray(np.uint8(sliced_array))
            face_image = face_image.convert("RGB")
            face_image.save(target)
            series = create_face_series(key, c, DATA_CATEGORY, target)
            df_sample_face.loc[target.stem] = series

    df_sample_face.to_csv(DTFR / f"df_{DATA_CATEGORY}_face.csv", index_label="uuid")

    ### Flatten directory
    flatten_directory(str(FACE), DATA_CATEGORY)
