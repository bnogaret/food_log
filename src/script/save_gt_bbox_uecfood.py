#! /usr/bin/env python3
import os

import pandas as pd

import init_path

from thesis_lib.utils_uecfood import read_bb_info_txt
from thesis_lib.io import save_object

import constants as const


def get_ground_truth_bbox():
    data = []

    for d in os.listdir(const.PATH_TO_ROOT_UECFOOD256):
        directory = os.path.join(const.PATH_TO_ROOT_UECFOOD256, d)
        print(directory)
        if os.path.isdir(directory):
            read_bb_info_txt(directory + "/bb_info.txt", data)

    # Transform the list into a pandas dataframe
    df = pd.DataFrame(data, columns=['_name', '_x1', '_y1', '_x2', '_y2', '_cat', '_abs_path'])
    
    # new_col = np.zeros((len(df)), dtype=bool)
    df['_multi_item'] = False

    print(df.head())
    print(df.head(1)._abs_path[0])
    print(df.describe())
    print(df.dtypes)
    print(df._cat.unique())
    print(len(df._cat.unique()))

    return df

def get_multiple_foot_image():
    path = "/panfs/storage/home/s242635/project/data/UECFOOD100/" + "/multiple_food.txt"
    
    data = []
    with open(path, 'r') as f:
        f.readline() # skip the first line that have the column names
        for line in f:
            split_line = line.split()
            split_line = [int(i) for i in split_line]
            
            image_name = split_line[0]
            
            for category in split_line[1:]:
                data.append([image_name, category])
    
    df = pd.DataFrame(data, columns=['_name', '_cat'])
    
    print(df.head())
    print(df.describe())
    print(df.dtypes)
    
    return df

if __name__ == "__main__":
    df = get_ground_truth_bbox()
    # print(df)
    df = get_multiple_foot_image()
    # save_object(df, const.PICKLE_FILENAME, overwrite=True)

