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

def read_bb_info_txt(path, array):
    """
    Read the bb_info.txt to get the rectangle coordinates and its class.
    Append this information into the list 'array' (thus, **the array is
    MODIFIED**).

    The structure of the appending values is:

    - first column (int): the file name without jpg
    - second (int) and third columns (int): coordinate of one of the corner
    - fourth (int) and fifth columns (int): coordinate of the opposte corner
    - sixth column (int): category of the bbox (directory name)
    - seventh column (str): absolute path to the image

    Parameters
    ----------
    path: str
        path to the file bb_info.txt file.
    array: list, modified
        list to append the different bounding boxes
    """
    # Get the label from the path (name of the directory containing the file)
    directory = os.path.dirname(path)
    label = os.path.split(directory)[1]

    with open(path, 'r') as f:
        f.readline() # skip the first line that have the column names
        for line in f:
            # print(line)
            split_line = line.split()

            split_line.append(label)
            data = [int(i) for i in split_line]
            
            data.append(os.path.abspath(directory + "/" + split_line[0] + ".jpg"))
            array.append(data)

def get_multiple_food_image(root_path):
    """
    Parse the multiple_food.txt file and return a dataframe corresponding to it.
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


def get_ground_truth_bbox(root_path, verbose=True):
    """
    Parse all the bb_info.txt and combine it with the multiple food image dataframe.
    
    Parameters
    ----------
    root_path: str
        Path to the UECFOOD 256 or 100 root directory
    
    Returns
    -------
    :class:`pandas.Dataframe`
        A pandas dataframe with:

        - '_img_name' (str): filename of the image
        -  '_x1' (int), '_y1' (int), '_x2' (int), '_y2' (int): coordinate of the bbox
        - '_cat' (int): category
        - '_abs_path' (str): absolute path to the image
    """
    data = []
    
    for entry in os.scandir(root_path):
        if entry.is_dir(follow_symlinks=False):
            read_bb_info_txt(entry.path + "/bb_info.txt", data)

    # Transform the list into a pandas dataframe
    df = pd.DataFrame(data, columns=['_img_name', '_x1', '_y1', '_x2', '_y2', '_cat', '_abs_path'])
    
    # Add a new column
    df['_multi_item'] = False
    
    if verbose:
        print(df.head())
        print(df.describe())
        print(df.dtypes)
        print(df.shape)
        print(df._cat.unique())
        print(len(df._cat.unique()))
    
    df_multiple_item = get_multiple_food_image(root_path)
    
    if verbose:
        print(df_multiple_item.head())
        print(df_multiple_item.dtypes)
        print(df_multiple_item.shape)
    
    df.ix[df._img_name.isin(df_multiple_item._img_name) & df._cat.isin(df_multiple_item._cat), '_multi_item'] = True
    
    return df
