import os

import caffe
import numpy as np
from skimage.transform import resize


def get_trained_network(path_to_model, path_to_weights, image_size, channel=3):
    """
    Load and return a trained network.

    Parameters
    ----------
    path_to_model: str
        Path to the model, i.e. the .prototxt file containing the definition
        of each layer.
    path_to_weights: str
        Path to the trained values for the corresponding model.
    image_size: array-like of 2 int
        size of the picture
    channel: int, optionnal
        The number of channel

    Returns
    -------
    :class:`caffe.Net`
        Initialised neural network

    """
    if not os.path.exists(path_to_model):
        raise IOError('Model file %s was not found!' % path_to_model)
    if not os.path.exists(path_to_weights):
        raise IOError('Trained network %s was not found!' % path_to_weights)

    net = caffe.Net(path_to_model,      # defines the structure of the model
                    path_to_weights,    # contains the trained weights
                    caffe.TEST)         # use test mode (e.g., don't perform dropout)

    net.blobs['data'].reshape(1,			        # batch size
                              channel,			    # number of channel
                              image_size[0],        # image size
                              image_size[1])

    return net


def transform_rgb_image(image, input_shape, mean_bgr_value):
    """
    Transform an image to make it compatible with the input of the CNN:
    
    - configured to take images in BGR format
    - values in the range [0, 255]
    - mean pixel of the trained dataset value subtracted from each channel
    - channel dimension is expected as the first (outermost) dimension
    
    It is a simplified version of :func:`get_transformer_rgb_image`.

    Parameters
    ----------
    image: array-like
        RGB image. **It is not modified**.
    input_shape: array-like of 2 int
        input shape of the network
    mean_bgr_value: :class:`numpy.ndarray` of 3 int
        Set the mean to subtract for centering the data.
        Must be in BGR order and in [0, 255]

    Returns
    -------
    :class:`numpy.ndarray`
        The transformed image

    References
    ----------
    https://github.com/BVLC/caffe/blob/master/python/caffe/io.py#L98
    """
    # Transform shape of the array: (3,) --> (3, 1, 1)
    mean = mean_bgr_value[:, np.newaxis, np.newaxis]
    # Convert image as a float
    img = image.astype(np.float32, copy=True)
    # Resize
    img = resize(img, input_shape, order=1)
    # move image channels to outermost dimension
    # Ex: shape: (224, 360, 3) --> (3, 224, 360)
    img = img.transpose((2, 0, 1))
    # swap channels from RGB to BGR
    img = img[(2, 1, 0),:,:]
    # Rescale image
    img *= 255
    # Substract the mean from each channel
    img -= mean
    
    return img

def get_transformer_rgb_image(input_shape, mean_bgr_value):
    """
    Return a caffe.Transformer class that modify an array to preprocess the
    input of a network.

    Parameters
    ----------
    input_shape: array-like of 2 int
        input shape of the network
    mean_bgr_value: :class:`numpy.ndarray` of 3 int
        Set the mean to subtract for centering the data.
        Must be in BGR order and in [0, 255]

    Returns
    -------
    :class:`caffe.io.Transformer`
        A ready-to-use transformer object

    References
    ----------
    https://github.com/BVLC/caffe/blob/master/python/caffe/io.py#L98
    """
    transformer = caffe.io.Transformer({'data': input_shape})

    transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
    transformer.set_mean('data', mean_bgr_value)        # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

    return transformer


def set_caffe_mode(mode="cpu"):
    """
    Set the mode (cpu or gpu) that caffe use for computation

    Parameters
    ----------
    mode: str, optional
        Either "cpu" or "gpu" to select the device
    """
    if mode == "cpu":
        caffe.set_mode_cpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()


