"""
File containing functions on bounding boxes (bbox).

A bouding box is represented by its coordinate as a :class:`numpy.ndarray` :math:`(x_0, y_0, x_1, y_1)`:

- :math:`(x_0, y_0)`: coordinate of one of the point
- :math:`(x_1, y_1)`: coordinate of the opposite point

"""
import numpy as np


def get_coordinate_resized_rectangles(base_shape, resized_shape, rectangles):
    """
    Update the rectangle's coordinates according to the resize of the image (points representing the rectangle).

    Parameters
    ---------
    base_shape: array-like of 2
        Size of the picture
    resized_shape: array-like of 2
        New size of the picture
    rectangles: array-like of 4
        List of rectangle coordinates to resized.

    Returns
    -------
    array-like of 4 floats
        Resized coordinates
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

    If the two boxes don't intersect, it returns 0.

    Parameters
    ----------
    bbox1: array-like of 4
        Coordinate of the one the first bbox

    bbox2: array-like of 4
        Coordinate of the one the second bbox

    Return
    ------
    number
        Intersection area

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

    Parameters
    ----------
    bbox: array-like of 4
        Coordinate of the bbox.

    Returns
    -------
    int
        Area of the bbox
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_overlap_ratio_bbox(bbox1, bbox2):
    """
    Compute the overlap ratio of two bounding boxes (intersection over union).

    Parameters
    ----------
    bbox1: array-like of 4
        Coordinate of the the first bbox

    bbox2: array-like of 4
        Coordinate of the the second bbox

    Returns
    -------
    float
        Overlap ratio

    References
    ----------
    http://uk.mathworks.com/help/vision/ref/bboxoverlapratio.html
    """
    intersection = get_intersection_bbox(bbox1, bbox2)
    union = get_area_bbox(bbox1) + get_area_bbox(bbox2) - intersection
    return intersection / union


def get_correct_bbox(ground_truth_bbox, predicted_bbox, threshold=0.5):
    """
    Compute the accuracy, precision and recall of the object detection of an
    image using the overlap (or intersection over union) metric (as defined in
    "The PASCAL Visual Object Classes Challenge 2012").
    It takes the list of predicted and ground truth bounding boxes to compute
    this three measures.

    WARNING: If more than one predicted bbox is true for a gt bbox, it will 
    only count one of them as true (keep the one with the highest overlap).

    Parameters
    ----------
    ground_truth_bbox: array-like of 4
        list of bbox coordinates
    predicted_bbox: array-like of 4
        list of bbox coordinates
    threshold : float, optional
        Ratio of area overlap to be considered as correct detection.
        Must be in [0, 1].

    Returns
    -------
    :class:`np.ndarray` of bbox coordinate
        The correct predicted bbox
    :class:`np.ndarray` of bbox coordinate
        The found ground truth bbox
    :class:`np.ndarray` of 3 float values:
        Metric results: accuracy, precision and recall values

    References
    ----------
    https://en.wikipedia.org/wiki/Precision_and_recall
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf
    """
    # Available ground-truth bbox
    gt = np.array(ground_truth_bbox)
    
    correct_predictions = []
    found_gt_bboxes = []
    
    for p in predicted_bbox:
        # grab the coordinates of the bounding boxes
        x1 = gt[:, 0]
        y1 = gt[:, 1]
        x2 = gt[:, 2]
        y2 = gt[:, 3]
        
        # compute the intersection and union
        # max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1)))
        intersection = np.maximum(0, np.minimum(x2, p[2]) - np.maximum(x1, p[0])) * \
                       np.maximum(0, np.minimum(y2, p[3]), - np.maximum(y1, p[1]))
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        area_p = (p[2] - p [0] + 1) * (p[3] - p[1] + 1)
        
        union = area + area_p - intersection
        
        # compute intersection over union
        overlap = intersection / union
        
        # get index of the highest value of overlap
        idx_max = np.argmax(overlap)
        
        if overlap[idx_max] > threshold:
            # The current prediction is correct
            correct_predictions.append(p)
            found_gt_bboxes.append(gt[idx_max].copy())
            # gt = np.delete(gt, idx_max, axis=0) # Delete the index from the available ground truth bboxes
            gt[idx_max] = 0 
        
        if gt.size == 0: # No more ground truth available, we can leave the loop
            break
    
    correct = len(correct_predictions)
    accuracy = correct / (correct + len(predicted_bbox) - correct + len(ground_truth_bbox) - correct)
    precision = correct / len(predicted_bbox)
    recall = correct / len(ground_truth_bbox)
    
    return np.asarray(correct_predictions).astype(np.int), \
           np.asarray(found_gt_bboxes).astype(np.int), \
           np.asarray([accuracy, precision, recall]).astype(np.float32)

def overlapping_suppression(boxes, confidence=None, overlap_threshold=0.4):
    """
    Delete redundant, overlapping bounding boxes.

    When two boxes are overlapping, it keeps the box with :

    - the strongest confidence if a confidence array is given
    - the highest y-coordinate of the second point (the 4th coordinate)

    Parameters
    ----------
    boxes: array-like of 4
        Array of bbox coordinates
    confidence: array-like, optional
        Array of confidence value. It is used to select the "best" bbox.
    overlap_threshold: float, optional
        Ratio to consider two bboxes to be overlapping.
        typical value: between 0.3 and 0.5

    Returns
    -------
    Numpy array of int
        The remaining boxes' coordinates

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

    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the index by either the confidence or the y-coordinate of the second point
    if confidence is None:
        idxs = np.argsort(y2)
        # idxs = np.argsort(y2)[::-1] for inverse
    else:
        idxs = np.argsort(confidence)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
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
        # an overlap > threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
