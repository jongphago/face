import os
from pathlib import Path
from typing import List
from PIL import Image
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpt.path import get_face_image_path_from_series as get_path

RANGE_TO_MEDIAN = {
    "a": (1 + 6) / 2,
    "b": (7 + 12) / 2,
    "c": (13 + 19) / 2,
    "d": (20 + 30) / 2,
    "e": (31 + 45) / 2,
    "f": (46 + 55) / 2,
    "g": (56 + 66) / 2,
    "h": (67 + 80) / 2,
    "above": 90,
}
AGE_GROUPS = ["a", "b", "c", "d", "e", "f", "g", "h", "above"]
GROUP_TO_INDEX = {group: index for index, group in enumerate(AGE_GROUPS)}


def age_to_age_groups(age):
    if age <= 6:
        return "a"
    if age <= 12:
        return "b"
    if age <= 19:
        return "c"
    if age <= 30:
        return "d"
    if age <= 45:
        return "e"
    if age <= 55:
        return "f"
    if age <= 66:
        return "g"
    if age <= 80:
        return "h"
    return "above"


def set_ax_locator(ax, size):
    width, height = size

    xlocator = ticker.FixedLocator([0, width - 1])
    ylocator = ticker.FixedLocator([0, height - 1])
    minor_locator = ticker.AutoLocator()

    ax.xaxis.set_major_locator(xlocator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_ticks_position("top")

    ax.yaxis.set_major_locator(ylocator)
    ax.yaxis.set_minor_locator(minor_locator)
    return ax


def crop_faces(
    key: str, value: List[List[str]], image_path: str, visualize: bool = False
) -> List[np.ndarray]:
    """이미지에서 정답 라벨을 따라 얼굴을 자르고 넘파이 배열로 반환합니다.
    파일 키와 바운딩박스 정답 라벨 이미지 경로를 파라미터로 받습니다. 이미지를 넘파이 배열로 변환 하고 배열을 슬라이싱 합니다.
    하나의 이미지에는 1개 또는 2개 이상의 얼굴이 있습니다.
    `visualize` 파라미터를 `True`로 설정하면 자른 얼굴을 시각화 합니다.

    Args:
        key (str): 이미지 파일의 이름을 구분하는 키
        value (List[List[str]]): 클래스와 바운딩박스`(x, y, w, h)`정보
        image_path (str): 키에 대응하는 이미지 경로
        visualize (bool, optional): 자른 얼굴 이미지 시각화 여부. Defaults to False.

    Raises:
        ValueError: 자른 얼굴의 폭과 너비 길이를 비교하고, 길이가 다르면 에러가 발생합니다. 길이가 다른 경우 정답 바운딩 박스가 잘못되어 있을수 있습니다.

    Returns:
        List[np.ndarray]: 크롭한 얼굴을 넘파이 배열로 담은 리스트
    """
    raw_pil_image = Image.open(image_path)
    raw_array = np.asarray(raw_pil_image)
    faces_list = []

    if visualize:
        fig, axs = plt.subplots(
            ncols=len(value),
            figsize=(4 * len(value), 4),
            gridspec_kw={"wspace": 0.1},
            constrained_layout=True,
        )
        fig.suptitle(key)
    else:
        axs = np.empty(len(value))

    for (id, *bbox), ax in zip(value, axs):
        x, y, width, height = list(map(lambda x: abs(int(x)), bbox))
        if not width == height:
            raise ValueError
        sliced_array = raw_array[y : y + height, x : x + width, :]
        faces_list.append(sliced_array)

        if visualize:
            ax.imshow(sliced_array)
            ax.set_xlabel(id)
            ax = set_ax_locator(ax, (width, height))

    return faces_list


def flatten_directory(root_path: str, sub_dir_name: str) -> None:
    """`root_path/sub_dir_name`경로에 있는 자른 얼굴 이미지를 id를 기준으로 폴더를 구조화 합니다.

    Args:
        root_path (str): 자른 얼굴 이미지의 루트 경로
        sub_dir_name (str): 데이터 카테고리 구분 경로
    """
    face_image_root = Path(root_path)
    os.makedirs(face_image_root / f"training_{sub_dir_name}", exist_ok=True)
    for src, dirs, files in os.walk(face_image_root / f"{sub_dir_name}"):
        if files:
            dst = face_image_root / f"training_{sub_dir_name}/{os.path.basename(src)}"
            os.symlink(src, dst)


def init_df_sample_face() -> pd.DataFrame:
    df_sample_face = pd.DataFrame(
        columns=[
            "key",
            "data_category",
            "folder_name",
            "family_id",
            "personal_id",
        ]
    )
    return df_sample_face


def create_face_series(key, c, DATA_CATEGORY, target) -> pd.Series:
    return pd.Series(
        {
            "key": key,
            "data_category": DATA_CATEGORY,
            "folder_name": target.parents[2].name,
            "family_id": key[:5],
            "personal_id": c,
        }
    )


def join_face_df(DTFR, data_category="aihub_sample"):
    face_path = DTFR / f"df_{data_category}_face.csv"
    file_path = DTFR / f"df_{data_category}_file.csv"
    _file = pd.read_csv(face_path, index_col="key", dtype={"folder_name": object})
    _face = pd.read_csv(file_path, index_col="key")
    face = _face.join(_file, on="key", how="right")
    face.loc[:, "path"] = face.apply(lambda x: str(get_path(x)), axis=1)
    face.loc[:, "target"] = face.apply(
        lambda x: f"{x.family_id}-{x.personal_id}", axis=1
    )
    face = face.sort_values(["family_id", "personal_id", "category", "option"])
    face = face.reset_index().set_index("uuid")

    # Age
    age_df = pd.read_csv(DTFR / "df_aihub_ages.csv", index_col=0)
    face = face.join(age_df, on="target")

    # Age group
    ages = face[face.category == "Age"]
    age_group_df = (
        ages.key.str.split("_").map(lambda x: x[-1][0]).to_frame(name="age_group")
    )
    face = face.join(age_group_df)

    # Update age
    face.loc[age_group_df.index, "age"] = None
    
    # fill age
    is_age_null = face['age'].isnull()
    range_to_age = face.loc[is_age_null, 'age_group'].map(lambda x: RANGE_TO_MEDIAN[x])
    face.loc[is_age_null, 'age'] = range_to_age
    
    # fill age group
    is_age_group_null = face['age_group'].isnull()
    age_groups = face.loc[is_age_group_null, 'age'].map(lambda x: age_to_age_groups(int(x)))
    face.loc[is_age_group_null, 'age_group'] = age_groups
    face.age = face.age.astype(int)

    return face
