"""
File containing functions on bounding boxes (bbox)
"""

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


def get_precision_recall_bbox(ground_truth_bbox, predicted_bbox, threshold=0.5):
    """
    Compute the recall and precision of the detection for an image.
    It takes the list of predicted and ground truth bounding boxes to compute this two measures.
    It may count several predicted bounding boxes as correct.
    
    Parameters
    ----------
    ground_truth_bbox: list of arrays of 4 elements
    predicted_bbox: list of arrays of 4 elements
    threshold: ratio of area overlap to be considered as correct detection
    
    Return
    ------
    precision, recall: two floats numbers
    """
    correct = 0
    for gt in ground_truth_bbox:
        for p in predicted_bbox:
            if (get_overlap_ratio_bbox(gt, p) > threshold):
                correct += 1
                # break
    precision = correct / len(predicted_bbox)
    recall = correct / len(ground_truth_bbox)
    return precision, recall