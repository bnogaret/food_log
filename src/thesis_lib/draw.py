import numpy as np
from skimage.draw import polygon_perimeter, polygon


def draw_bounding_boxes(image, bbox, color=0, copy=False):
    """
    Draw rectangles on the image, using the representation with two points (pt1_x, pt1_y, pt2_x, pt2_y).
    If copy is true, it return a deep copy, otherwise it modifies directly the parameter 'image'.
    
    Parameter
    --------
    image: 2D or 3D array
    bbox: list of rectangles to draw on the image.
    color: integer used to draw the rectangle
    copy: boolean
    
    Return
    ------
    2D array if copy is true
    """
    if copy:
        img = image.copy()
    else:
        img = image
    
    for bb in bbox:
        coor = np.array((
            (bb[0], bb[1]),
            (bb[0], bb[3]),
            (bb[2], bb[3]),
            (bb[2], bb[1])
        ))
        rr, cc = polygon_perimeter(coor[:, 1], coor[:, 0], img.shape)
        img[rr, cc] = color
    
    return img
