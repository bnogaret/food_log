import numpy as np

def get_sub_image_from_rectangle(image, rectangle, copy=False):
    """
    Return the sub-array of the image that is included inside rectangle.
    If copy is true, it returns a new array (sub_image.flags['OWNDATA'] is true), otherwise it returns a view.
    """
    # print(rectangle)
    rows, cols, chan = image.shape
    if copy:
        return np.copy(image[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]])
    else:
        return image[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]