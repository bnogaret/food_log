import numpy as np
from skimage.draw import polygon_perimeter, polygon


def rectangle_perimeter(image, pt1, pt2, color=0, copy=False):
    """
    Draw a rectangle on the image, using the representation with two points.
    If copy is true, it return a deep copy, otherwise it modifies directly the parameter 'image'.
    
    Parameter
    --------
    image: 2D or 3D array
    pt1: 2D coordinate of one of the corner
    pt2: 2D coordinate of the opposite corner of pt1
    color: integer used to draw the rectangle
    copy: boolean
    
    Return
    ------
    2D array if copy is true
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
