#! /usr/bin/env python3
import os
import argparse

import pandas as pd

import init_path

from thesis_lib.uecfood import read_bb_info_txt
from thesis_lib.io import save_object

import constants as const

def argument_parser():
    parser = argparse.ArgumentParser(description="Create pandas dataframe containing category, coordinate and path of the bounding box")
    parser.add_argument('-d',
                        '--dataset',
                        help='Choose the dataset between : ' + \
                             '"100": UEC-FOOD100 ' + \
                             '"256": UEC-FOOD256',
                        choices=['256', '100'],
                        default='256',
                        type=str)
    
    args = parser.parse_args()
    return args


def get_multiple_food_image(root_path):
    """
    Parse the multiple_food.txt file and return a dataframe.
    """
    
    data = []
    with open(root_path + "/multiple_food.txt", 'r') as f:
        f.readline() # skip the first line that have the column names
        for line in f:
            split_line = line.split()
            split_line = [int(i) for i in split_line]
            
            image_name = split_line[0]
            
            for category in split_line[1:]:
                data.append([image_name, category])
    
    df = pd.DataFrame(data, columns=['_img_name', '_cat'])

    return df


def get_ground_truth_bbox(root_path):
    """
    Parse all the bb_info.txt and combine it with the multiple food image dataframe.
    """
    data = []
    
    for entry in os.scandir(const.PATH_TO_ROOT_UECFOOD256):
        if entry.is_dir(follow_symlinks=False):
            print(entry.name)
            read_bb_info_txt(entry.path + "/bb_info.txt", data)

    # Transform the list into a pandas dataframe
    df = pd.DataFrame(data, columns=['_img_name', '_x1', '_y1', '_x2', '_y2', '_cat', '_abs_path'])
    
    # Add a new column
    df['_multi_item'] = False

    print(df.head())
    print(df.describe())
    print(df.dtypes)
    print(df._cat.unique())
    print(len(df._cat.unique()))
    
    df_multiple_item = get_multiple_food_image(root_path)
    
    df.ix[df._img_name.isin(df_multiple_item._img_name) & df._cat.isin(df_multiple_item._cat), '_multi_item'] = True
    
    print(df.loc[df._img_name == 199])
    print(df.loc[df._img_name == 4004])
    print(df.loc[df._img_name == 5109])
    print(df.loc[df._img_name == 97])
    print(df.loc[df._img_name == 14930])
    
    return df

def main():
    args = argument_parser()
    
    print(args)
    
    arg_dataset = {
        '256'   : (const.PATH_TO_ROOT_UECFOOD256, const.PICKLE_FILENAME_256_GT_BBOX),
        '100'   : (const.PATH_TO_ROOT_UECFOOD100, const.PICKLE_FILENAME_100_GT_BBOX),
    }
    
    df = get_ground_truth_bbox(arg_dataset[args.dataset][0])
    # print(df)
    
    save_object(df, arg_dataset[args.dataset][1], overwrite=True)


if __name__ == "__main__":
    main()

