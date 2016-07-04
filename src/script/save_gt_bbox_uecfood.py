#! /usr/bin/env python3
import os
import argparse

import pandas as pd

import init_path

from thesis_lib.uecfood import read_bb_info_txt, get_ground_truth_bbox
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

