from warnings import warn

import numpy as np
import itertools
from skimage.feature import local_binary_pattern


def color_histogram(image, bins=40, distribution="marginal", ranges=(0, 256), eps=1e-7, normalization=True):
    """
    Get the color histogram of the image. It can either work on independantly
    for each channel (marginal distribution) or by combination of 2 channels
    (joint distribution).
    
    .. warning::
        If joint is used, be careful not to have too many channels / a 
        lot of bins.

    Parameters
    ----------
    image: array-like
        a 2d or 3D array of double or uint8 corresponding to an image
    bins: int, optional
        number of bins of the histogram
    distribution: str, optional
        either 'marginal' or 'joint'
        Compute marginal histogram for each channel or joint histogram for each 
        2D combination of channel. If 'joint' is used, be careful not to put a 
        too big 'bins' value and / or execute it on too many channels.
    range: tuple of 2 numbers or :class:`numpy.ndarray` of 2 numbers, optional
        range of value of the channel.
        For a joint histogram
    normalization: bool, optional
        normalize the histogram (put its value between in [0, 1])

    Returns
    -------
    :class:`numpy.ndarray`
        Color histogram of size:
        - ((bins + 2) * channel) for 'marginal' distribution
        - (bins * bins * :math:`\dbinom{number_of_channels}{2}`) for 'joint' distribution

    References
    ---------
    http://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
    https://en.wikipedia.org/wiki/Combination
    """
    # Get the number of channels of the image
    # http://stackoverflow.com/a/19063058
    nb_chan =  image.shape[2] if len(image.shape) == 3 else 1
    
    features = []
    
    if distribution == "joint" and nb_chan <= 1:
        warn("Not enough channel to execute a joint histogram (at least 2 channels are required).")
        distribution = "marginal"
    
    if distribution == "marginal":
        for i in range(nb_chan):
            # create a histogram for the current channel and
            (hist, _) = np.histogram(image[:, :, i],
                                     bins=bins,
                                     range=ranges)

            # normalize the histogram
            if normalization:
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)
            
            # concatenate the resulting histograms for each channel
            features.append(hist)
    
    elif distribution == "joint":
        # iter over all the 2 elements combination of channels
        for i, j in itertools.combinations(range(nb_chan), 2):
            (hist, *_) = np.histogram2d(image[:, :, i].flatten(), # select the i-th channel
                                        image[:, :, j].flatten(), # select the j-th channel
                                        bins=bins,
                                        range=ranges)
            
            hist = hist.flatten()
            
            # normalize the histogram
            if normalization:
                hist = hist.astype("float")
                hist /= (hist.sum() + eps) 
            
            features.append(hist)

    return np.array(features).flatten()


def local_binary_pattern_histogram(image, numPoints=18, radius=8, eps=1e-7, normalization=True):
    """
    Compute the local binary pattern histogram for an image. The function is a wrapper
    above :func:`skimage.feature.local_binary_pattern`.

    The number of bins is: numPoints + 3.
    If the histogram is not normalized, its values are in [0, numPoints + 1].

    Parameters
    ----------
    image: 2D array-like
        *Gray-level* image
    numPoints: int, optional
        Number of circularly symmetric neighbour set points (quantization of the angular space).
    radius: int, optional
        radius of circle used in the LBP algorithm
    normalization: bool, optional
        normalize the histogram (put its value between in [0, 1])

    Returns
    -------
    :class:`numpy.ndarray` of size numPoints + 3
        LBP histogram

    References
    ----------
    http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern
    """
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = local_binary_pattern(image,
                               numPoints,
                               radius,
                               method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))

    # normalize the histogram
    if normalization:
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist
