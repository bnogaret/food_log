import os

import caffe


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
    
    Return
    -------
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


def get_transformer_rgb_image(input_shape, mean_value, input_scale=255):
    """
    Return a caffe.Transformer class that modify an array to preprocess the 
    input of a network.
    
    Parameters
    ----------
    input_shape: array-like of 2 int
        input shape of the network
    mean_value: array-like of 3 numbers
        Set the mean to subtract for centering the data.
        Must have the same scale as the one given by input scale.
    input_scale: number, optional
        set the scale of the input_blob (realised before any other preprocessing).
        Python represent images with value in [0, 1], yet, some caffe model 
        expect an other scale (example: [0, 255]).
        input_blob = initial_input * input_scale
    
    Return
    -------
    Ready to use transformer
    
    References
    ----------
    https://github.com/BVLC/caffe/blob/master/python/caffe/io.py#L98
    """
    transformer = caffe.io.Transformer({'data': input_shape})

    transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
    transformer.set_mean('data', mean_value)    # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', input_scale)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR
    
    return transformer


def set_caffe_mode(mode="cpu"):
    """
    Set the mode (cpu or gpu) that caffe use for computation
    
    Parameter
    ---------
    mode: str, optional
        Either "cpu" or "gpu" to select the device
    """
    if mode == "cpu":
        caffe.set_mode_cpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()


