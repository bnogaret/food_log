import os
import glob

import pandas as pd


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

