import abc

import numpy as np
from skimage import img_as_float

from .cnn import transform_rgb_image, get_trained_network
from .bbox import overlapping_suppression


__all__ = ['BaseSegmenter',
           'CnnSegmenter']

class BaseSegmenter(metaclass=abc.ABCMeta):
    """
    Abstract class common for my image segmenters.
    """
    
    @abc.abstractmethod
    def get_bbox(self, image):
        """
        Extract and return the bbox from the image
        
        Parameters
        ----------
        image: :class:`numpy.ndarray`
            **RGB** image
        
        Returns
        -------
        :class:`numpy.ndarray`
            Array of bbox coordinates (4 int values)
        """
        return


class CnnSegmenter(BaseSegmenter):
    """
    Use a pre-trained CNN to segment food items from an image (caffe CNN).
    
    Post-process: scale the data (if scale == True)
    """
    
    def __init__(self, path_to_model_def, path_to_model_weights, path_to_bbox_coordinates, mean_bgr, image_size=(400, 400), threshold_net=0.1, threshold_overlap=0.5):
        """
        Parameters
        ----------
        path_to_model_def: str
            Path to the model, i.e. the .prototxt file containing the definition
            of each layer.
        path_to_model_weights: str
            Path to the trained values for the corresponding model.
        path_to_bbox_coordinates: str
            Path to the center100.txt file containing the bbox coordinates
        mean_bgr: :class:`numpy.ndarray` of 3 int
            Set the mean to subtract for centering the data.
            Must be in BGR order and in [0, 255]
        image_size: array-like of 2 int
            size of the input picture of the net
        threshold_net: float
            Threshold net value. Must be in [0, 1]
        threshold_overlap: float
            Threshold to consider two overlapping bounding boxes to be the same.
            Must be in [0, 1] (usually between O.3 and O.5)
        """
        self.mean_bgr = mean_bgr
        self.image_size = image_size
        self.net = get_trained_network(path_to_model_def,
                                       path_to_model_weights,
                                       self.image_size)
        self.cnn_bbox_coordinates = np.loadtxt(path_to_bbox_coordinates,
                                               np.float,
                                               delimiter=',')
        self.threshold_net = threshold_net
        self.threshold_overlap = threshold_overlap
        
    
    def get_bbox(self, image):
        transformed_image = img_as_float(image).astype(np.float32)
        
        transformed_image = transform_rgb_image(transformed_image,
                                                self.image_size,
                                                self.mean_bgr)
        # Put the image as input of the network
        self.net.blobs['data'].data[...] = transformed_image

        # perform classification
        output = self.net.forward()
        
        prob = output['prob'][0]
                
        # Get the predicted bounding box
        predicted_index = np.where(prob > self.threshold_net)[0]
        if predicted_index.size == 0:
            predicted_index = np.where(prob == prob.max())[0]
        
        # Get the coordinate in image size
        predicted_boxed = self.cnn_bbox_coordinates[predicted_index, :]
        predicted_boxed[:, ::2] *= image.shape[1]
        predicted_boxed[:, 1::2] *= image.shape[0]
        predicted_boxed = np.ceil(predicted_boxed)
        
        # print(predicted_boxed)
        
        # Delete possible bbox duplications
        boxes = overlapping_suppression(predicted_boxed, prob[predicted_index], self.threshold_overlap)
        
        return boxes


