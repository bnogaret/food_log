"""
File containing functions on bounding boxes (bbox)
"""

import numpy as np

def get_coordinate_resized_rectangles(base_shape, resized_shape, rectangles):
    """
    Update the rectangle's coordinates according to the resize of the image (points representing the rectangle).
    
    Parameter
    ---------
    base_shape: size of the picture (array-like of 2 elements)
    resized_shape: new size of the picture (array-like of 2 elements)
    rectangles: list of rectangle coordinates
        (rectangle coordinate = array of [x_0, y_0, x_1, y_1] with x and y of opposite points).
    
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


def get_intersection_bbox(bbox1, bbox2):
    """
    Compute the intersection area between two bounding boxes / rectangles.
    
    References
    ----------
    http://uk.mathworks.com/help/vision/ref/bboxoverlapratio.html
    https://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap
    """
    # max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1)))
    return max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])) * \
            max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))

def get_area_bbox(bbox):
    """
    Compute the area of a bounding box.
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    
def get_overlap_ratio_bbox(bbox1, bbox2):
    """
    Compute the overlap ratio of two bounding boxes (intersection / union)
    """
    intersection = get_intersection_bbox(bbox1, bbox2)
    union = get_area_bbox(bbox1) + get_area_bbox(bbox2) - intersection
    return intersection / union


def get_accuracy_bbox(ground_truth_bbox, predicted_bbox, threshold=0.5):
    """
    Compute the accuracy, precision and recall of the object detection of an 
    image using the overlap (or intersection over union) metric (as defined in 
    "The PASCAL Visual Object Classes Challenge 2012").
    It takes the list of predicted and ground truth bounding boxes to compute 
    this three measures.
    
    WARNING: It may count several predicted bounding boxes as correct.
    
    Parameters
    ----------
    ground_truth_bbox: list of arrays of 4 int elements
    predicted_bbox: list of arrays of 4 int elements
    threshold (float): ratio of area overlap to be considered as correct detection
    
    Return
    ------
    accuracy, precision, recall: three floats numbers
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Precision_and_recall
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf
    """
    correct = 0
    for gt in ground_truth_bbox:
        for p in predicted_bbox:
            if (get_overlap_ratio_bbox(gt, p) > threshold):
                correct += 1
    
    accuracy = correct / (correct + len(predicted_bbox) - correct + len(ground_truth_bbox) - correct)
    precision = correct / len(predicted_bbox)
    recall = correct / len(ground_truth_bbox)
    return accuracy, precision, recall


def non_maxima_suppression(boxes, confidence, overlapThresh=0.4):
    """
    Delete redundant, overlapping bounding boxes, keeping the boxes with the strongest confidence.
    
    Parameters
    ----------
    boxes: numpy array of bbox
    overlapThresh: typical value: between 0.3 and 0.5
    
    Return
    -------
    Numpy array (int) of the remaining boxes
    
    References
    ----------
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    # initialize the list of picked indexes	
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box / confidence score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # idxs = np.argsort(y2)
    """
    if confidence is None:
        idxs = np.argsort(x1)#[::-1]
    else:
        idxs = np.argsort(confidence)
    """
    idxs = np.argsort(confidence)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        # last = len(idxs) - 1
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
