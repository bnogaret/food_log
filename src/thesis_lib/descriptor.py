import abc
import warnings

import cv2
import numpy as np

from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2gray, rgb2hsv
from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans

from .histogram import local_binary_pattern_histogram, color_histogram
from .measure import get_hu_moment_from_image
from .cnn import transform_rgb_image, get_trained_network


__all__ = ['BaseDescriptor',
           'HistogramMomentDescriptor',
           'BagOfWordsDescriptor',
           'CnnDescriptor']


class BaseDescriptor(metaclass=abc.ABCMeta):
    """
    Abstract class common for my image descriptors.
    """

    @abc.abstractmethod
    def get_feature(self, image):
        """
        Extract and return the feature from the image

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            **RGB** image

        Returns
        -------
        :class:`numpy.ndarray`
            Feature
        """
        return

    @abc.abstractmethod
    def post_process_data(self, data, target):
        """
        Post process the list of feature extracted by this class

        Parameters
        ----------
        data: list of :class:`numpy.ndarray`
            List of features
        target: list
            Labels
        Returns
        -------
        X :class:`numpy.ndarray`, y :class:`numpy.ndarray`
            Return the processed data and target
        """
        return


class HistogramMomentDescriptor(BaseDescriptor):
    """
    Histogram descriptor:

    - local binary pattern histogram
    - color histogram: joint histogram on H and S channels or marginal histogram for R, G and B channels
    - mean and var for each picture channel (RGB, gray, HS from HSV)
    - 7 hu moments

    Post-process: scale the data (if scale_data == True)
    """

    def __init__(self, bin_lbp = 48, bin_ch = 20, distribution='joint', scale_data=True):
        """
        Parameters
        ----------
        bin_lbph: int
            Number of bins for the LBP histogram
        bin_ch: int
            Number of bins for the color histogram
        distribution: str
            Joint or marginal
        scale: bool
            Whether or not scale the data in post-process
        """
        self.bin_lbp = bin_lbp
        self.bin_ch = bin_ch
        self.distribution = distribution
        self.scale_data = scale_data

    def get_feature(self, image):
        gray = rgb2gray(image)
        hsv = rgb2hsv(image)

        lbph = local_binary_pattern_histogram(gray, self.bin_lbp, 8)

        if self.distribution == "joint":
            ch = color_histogram(hsv[:,:,:2], self.bin_ch, ranges=((0,1),(0,1)), distribution=self.distribution)
        else:
            ch = color_histogram(image, self.bin_ch, ranges=(0,1), distribution=self.distribution)

        # Mean and variance of each RGB channel
        mean_rgb = np.mean(image, axis=(0,1))
        var_rgb = np.var(image, axis=(0,1))
        
        # Meand and var for grayscale
        mean_gray = np.mean(gray, axis=(0,1))
        var_gray = np.var(gray, axis=(0,1))
        
        # Mean and var for H and S
        mean_hs = np.mean(hsv[:,:,:2], axis=(0,1))
        var_hs = np.var(hsv[:,:,:2], axis=(0,1))
        
        # 7 Hu moments
        hu = get_hu_moment_from_image(gray)

        feature = np.hstack((lbph, ch, mean_rgb, var_rgb, mean_gray, var_gray, mean_hs, var_hs, hu))

        return feature

    def post_process_data(self, data, target):
        X = np.asarray(data)
        y = np.asarray(target)

        if self.scale_data:
            # scale the data: zero mean and unit variance
            X = scale(X)
        return X, y


class BagOfWordsDescriptor(BaseDescriptor):
    """
    Bag of words feature description.

    Post-process:

    - create the visual words using the KNN method
    - get histogram for each data
    - scale the data (if scale_data == True)
    """

    def __init__(self, image_size=(400, 400), vocabulary_size=1000, step_size=4, scale_data=True):
        """
        Parameters
        ----------
        image_size: tuple of 2 int
            Size of the image
        vocabulary_size: int
            Number of visual words
        step_size: int
            Size of the dense grid
        scale_data: bool
            Whether or not scale the data in post-process
        """
        self.kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, image_size[1], step_size)
                                                 for x in range(0, image_size[0], step_size)]
        self.descriptor = cv2.xfeatures2d.SIFT_create()
        self.vocabulary_size = vocabulary_size
        self.clustering = MiniBatchKMeans(n_clusters=self.vocabulary_size,
                                          verbose=0,
                                          init='k-means++',
                                          batch_size=2 * self.vocabulary_size,
                                          compute_labels=False,
                                          reassignment_ratio=0.0, # http://stackoverflow.com/a/23527049
                                          n_init=5,
                                          max_iter=50)
        self.scale_data = scale_data

    def get_feature(self, image):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gray = img_as_ubyte(rgb2gray(image))

        _, feature = self.descriptor.compute(gray, self.kp)

        # Root feature:
        eps = 1e-5 # not to divide by 0.0
        # L1 normalize
        feature = feature / (feature.sum(axis=1, keepdims=True) + eps)
        # take the square root
        feature = np.sqrt(feature)

        return feature

    def post_process_data(self, data, target):
        y = np.asarray(target)
        words = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for d in data:
                self.clustering.partial_fit(d)
            for d in data:
                words.append(self.clustering.predict(d))

        words = np.vstack(words)
        print("Cluster prediction: ", words.shape)

        X = []

        for word in words:
            histogram = np.bincount(word, minlength=self.vocabulary_size).astype(np.float)
            histogram /= max(histogram.sum(), 1)

            X.append(histogram.flatten())

        X = np.vstack(X)

        if self.scale_data:
            # scale the data: zero mean and unit variance
            X = scale(X)

        return X, y


class CnnDescriptor(BaseDescriptor):
    """
    Use a pre-trained CNN to describe an image (caffe CNN).

    Post-process: scale the data (if scale_data == True)
    """

    def __init__(self, layer_name, path_to_model_def, path_to_model_weights, mean_bgr, image_size=(400, 400), scale_data=True):
        """
        Parameters
        ----------
        layer_name: str
            Name of the layer to get the output result
        path_to_model_def: str
            Path to the model, i.e. the .prototxt file containing the definition
            of each layer.
        path_to_model_weights: str
            Path to the trained values for the corresponding model.
        mean_bgr: :class:`numpy.ndarray` of 3 int
            Set the mean to subtract for centering the data.
            Must be in BGR order and in [0, 255]
        image_size: array-like of 2 int
            size of the input picture of the net
        scale_data: bool
            Whether or not scale the data in post-process
        """
        self.layer_name = layer_name
        self.mean_bgr = mean_bgr
        self.image_size = image_size
        self.net = get_trained_network(path_to_model_def,
                                       path_to_model_weights,
                                       self.image_size)
        self.scale_data = scale_data

    def get_feature(self, image):
        transformed_image = img_as_float(image).astype(np.float32)

        transformed_image = transform_rgb_image(transformed_image,
                                                self.image_size,
                                                self.mean_bgr)
        # Put the image as input of the network
        self.net.blobs['data'].data[...] = transformed_image

        # perform classification
        output = self.net.forward()

        # Get value of the last layer before classification layers
        feature = self.net.blobs[self.layer_name].data.flatten()
        return feature

    def post_process_data(self, data, target):
        X = np.asarray(data)
        y = np.asarray(target)

        if self.scale_data:
            # scale the data: zero mean and unit variance
            X = scale(X)
        return X, y

