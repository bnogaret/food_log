"""
Contains useful functions for the UEFCFOOD256 dataset.
"""

import os


def read_bb_info_txt(path, array):
    """
    Read the bb_info.txt to get the rectangle coordinates and its class.
    Append this information into the list 'array' (thus, the array is obviously 
    MODIFIED).
    
    The structure of the appending value is:
    - first column (int): the image id (file name without jpg)
    - second (int) and third columns (int): coordinate of one of the corner
    - fourth (int) and fifth columns (int): coordinate of the opposte corner
    - sixth column (int): label / class of the bbox (directory name)
    
    Parameter
    ---------
    path: path to the file bb_info.txt file.
    array: list to append the different bounding boxes
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
