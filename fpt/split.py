import pandas as pd
from fpt.path import UTIL


def get_split_path(STAGE):
    return UTIL / "dataset" / "split" / f"{STAGE}_split.txt"


def write_split(STAGE, uuids):
    with open(get_split_path(STAGE), "w") as f:
        for uuid in uuids:
            f.write(uuid)
            f.write("\n")


def read_split(STAGE):
    with open(get_split_path(STAGE), "r") as f:
        out = f.readlines()
        face_uuids = [row.rstrip() for row in out]
    return face_uuids
