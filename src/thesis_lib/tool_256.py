"""
Contains functions for that are only applicable for the UEFCFOOD256 dataset.
"""

import os
import glob

import pandas as pd


def get_name_and_category(filename):
    """
    Return a dataframe containing the name and the global category for each
    label of the UEFC-FOOD-256 dataset.

    The 5 possible categories are from ChooseMyPlate and are:

    - fruit
    - protein: meat, egg, ...
    - vegetable
    - dairy
    - grain

    Parameters
    ----------
    filename: string
        path to the file containing the name and categories
        (modified version of category.txt from the dataset).

    Returns
    -------
    :class:`pandas.Dataframe`
        A pandas dataframe with:

        - _id: int
            index, starting from 1, corresponding to the label
        - _name: str
            name of the label
        - _category: category
            one of the five possible categories for a food item.

    References
    ----------
    https://en.wikipedia.org/wiki/Food_group
    http://www.choosemyplate.gov/
    """
    df = pd.read_csv(filename,
                     delimiter=r"\t|\s{2,}",
                     header=0,
                     names=["_id", "_name", "_category"],
                     engine="python")

    df = df.set_index('_id')

    df = df.drop('_category', axis=1) \
           .join(df._category.str.split(',', expand=True) \
                                 .stack().reset_index(drop=True, level=1) \
                                 .rename('_category'))

    df._category = df._category.astype("category",
                                       categories=["grain", "vegetable", "protein", "dairy", "fruit"],
                                       ordered=False)

    return df


def get_list_image_file(path_to_root_dir):
    """
    Return a dataframe containing the path to each **different** picture.
    Two pictures are considered the same if they have the same file name (image id).

    Parameters
    ----------
    path_to_root_dir: str
        Path to the root directory of the UEC FOOD 256 dataset.
    
    Returns
    -------
    :class:`pandas.Dataframe`
        A pandas dataframe with:

        - _image_id: int
            index, starting from 1, corresponding to the image id
        - _image_path: str
            absolute path to the image
    """
    dictionary = {}

    for filename in glob.iglob(path_to_root_dir + '/**/*.jpg', recursive=True):
        file_without_jpg = os.path.basename(filename).replace(".jpg", '')
        dictionary[int(file_without_jpg)] = os.path.abspath(filename)
    
    df_filepath = pd.DataFrame(list(dictionary.items()),
                               columns=['_image_id','_image_path'])
    
    return df_filepath


def read_bb_info_txt(path, array):
    """
    Read the bb_info.txt to get the rectangle coordinates and its class.
    Append this information into the list 'array' (thus, the array is obviously
    MODIFIED).

    The structure of the appending values is:

    - first column (int): the image id (file name without jpg)
    - second (int) and third columns (int): coordinate of one of the corner
    - fourth (int) and fifth columns (int): coordinate of the opposte corner
    - sixth column (int): label / class of the bbox (directory name)

    Parameters
    ----------
    path: str
        path to the file bb_info.txt file.
    array: list, modified
        list to append the different bounding boxes
    """
    # Get the label from the path (name of the directory containing the file)
    label = os.path.split(os.path.dirname(path))[1]

    with open(path, 'r') as f:
        f.readline() # skip the first line that have the column names
        for line in f:
            # print(line)
            split_line = line.split()

            split_line.append(label)
            array.append([int(i) for i in split_line])

