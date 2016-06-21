#! /usr/bin/env python3
import os

import pandas as pd

import init_path

from thesis_lib.tool_256 import read_bb_info_txt
from thesis_lib.io import save_object


PATH_CURRENT_DIRECTORY = os.path.dirname(__file__)
PATH_TO_UECFOOD256 = os.path.join(PATH_CURRENT_DIRECTORY, "..", "../data/UECFOOD256/")


def get_ground_truth_bbox():
    data = []

    for d in os.listdir(PATH_TO_UECFOOD256):
        directory = os.path.join(PATH_TO_UECFOOD256, d)
        print(directory)
        if os.path.isdir(directory):
            read_bb_info_txt(directory + "/bb_info.txt", data)

    # Transform the list into a pandas dataframe
    df = pd.DataFrame(data, columns=['image_id', 'x1', 'y1', 'x2', 'y2', 'label'])

    print(df.head())
    print(df.describe())
    print(df.label.unique())
    print(len(df.label.unique()))
    
    return df


if __name__ == "__main__":
    df = get_ground_truth_bbox()
    save_object(df, "256_gt_bbox", overwrite=True)

