import numpy as np
from skimage.feature import local_binary_pattern


def color_histogram(image, bins=40, ranges=(0, 256), eps=1e-7):
    """
    Get the color histogram for each channel of the image
    
    Parameters
    ----------
    bins: number of bins of the histogram
    range: range of value of the channel
    
    Returns
    -------
    output : (N, M) array
        LBP image.
    """
    # Get the number of channels of the image
    # http://stackoverflow.com/a/19063058
    nb_chan =  image.shape[2] if len(image.shape) == 3 else 1
    features = []
    # Get the number of channels of the image
    # http://stackoverflow.com/a/19063058
    # chans = cv2.split(image)
    for i in range(nb_chan):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        # hist = cv2.calcHist([image[:, :, i]], [0], None, histSize, ranges)
        (hist, _) = np.histogram(image[:, :, i],
                                 bins=bins,
                                 range=ranges)

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        features.append(hist)
    
    return np.array(features).flatten()


def local_binary_pattern_histogram(image, numPoints=18, radius=8, eps=1e-7):
    """
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
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist