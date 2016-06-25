import numpy as np


def get_sub_image_from_rectangle(image, rectangle, copy=False):
    """
    Return the sub-array of the image that is included inside the rectangle.
    If copy is true, it returns a new array (sub_image.flags['OWNDATA'] is true),
    otherwise it returns a view.

    Parameters
    ----------
    image: array-like
        a 2d array of double or uint8 corresponding to an image
    rectangle: array-like of 4 int
        coordinate of a rectangle: array of 4 values

    Returns
    -------
    Array-like
        Sub-image. It is either a view if copy is false, a new numpy array otherwise.
    """
    # print(rectangle)
    # rows, cols, chan = image.shape
    if copy:
        return np.copy(image[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]])
    else:
        return image[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
