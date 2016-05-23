import numpy as np
from skimage.draw import polygon_perimeter, polygon

def rectangle_perimeter(image, pt1, pt2, color=0, copy=False):
    """
    Return a new image (deep copy of image with the rectangle).
    """
    
    coor = np.array((
        pt1,
        (pt1[0], pt2[1]),
        pt2,
        (pt2[0], pt1[1])
    ))
    rr, cc = polygon_perimeter(coor[:, 1], coor[:, 0], image.shape)
    if copy:
        copy = image.copy()
        copy[rr, cc] = color
        return copy
    else:
        image[rr, cc] = color
        return image