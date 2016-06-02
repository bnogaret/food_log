import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def read_bb_info_txt(path, d, verbose = False):
    """
    Read the bb_info.txt to get the rectangle coordinates.
    Add the coordinates of the rectangles into the dictionnary d.
    The name of the file is used as the key, the value is a list of 4 integers separated by a space:
    - first and second value: coordinate of one of the corner
    - third and forth value: coordinate of the opposte corner
    
    Parameter
    ---------
    path: path to the file bb_info.txt file.
    d: dictionnary storing the rectangle coordinates
    """
    label = os.path.split(os.path.dirname(path))[1]
    with open(path, 'r') as f:
        f.readline() # skip the first line
        for line in f:
            # print(line)
            data = line.split()
            if data[0] in d:
                if verbose:
                    print("Multiple image for {}".format(data[0]))
                d[data[0]].append([int(i) for i in data[1:]])
            else:
                d[data[0]] = [[int(i) for i in data[1:]]]


def get_coordinate_resized_rectangles(base_shape, resized_shape, rectangles):
    """
    Update the rectangle's coordinates according to the resize of the image (points representing the rectangle).
    
    Parameter
    ---------
    base_shape: size of the picture (array-like of 2 elements)
    resized_shape: new size of the picture (array-like of 2 elements)
    rectangles: list of rectangle coordinates
        (rectangle coordinate = array of [x_0, y_0, x_1, y_1] with x and y of opposite size).
    
    Return:
    -------
    list of arrays
    """
    x_scale = resized_shape[0]/base_shape[1]
    y_scale = resized_shape[1]/base_shape[0]
    resized_rectangles = []
    for rectangle in rectangles:
        resized_rectangles.append([
                int(rectangle[0] * x_scale),
                int(rectangle[1] * y_scale),
                int(rectangle[2] * x_scale), 
                int(rectangle[3] * y_scale)
                ])
    return resized_rectangles


def plot_confusion_matrix(confusion_matrix, target_names, normalization=True, title='Confusion matrix', cmap=plt.cm.OrRd):
    """
    Plot the confusion matrix.
    
    Parameters
    ----------
    confusion matrix: 2D array (M * M)
    target_names: name of each element of the matrix
    normalization: normalization of the confusion matrix to get value in [0, 1] for each line
    titel: title of the plot
    cmap: color map of the confusion matrix
    
    References
    ----------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    """
    if normalization:
        # Normalize the confusion matrix by row
        # (i.e by the number of samples in each class)
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusion_matrix
    print("Confusion matrix")
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
