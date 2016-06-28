import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon_perimeter


def draw_bounding_boxes(image, bbox, color=0, copy=False):
    """
    Draw rectangles on the image, using the representation described in `bbox`.
    If copy is true, it return a deep copy, otherwise it modifies directly the parameter 'image'.

    Parameters
    ----------
    image: 2D or 3D array
    bbox: list of rectangles to draw on the image.
    color: integer used to draw the rectangle
    copy: boolean

    Returns
    -------
    Nothing ot 2D array-like
        Return an array if copy is true.
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


def display_arrays(data, filename=None, title="", normalize=True, padding_value=1):
    """
    Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) / (hieght, width, 3) thing in a grid of
    size approximately sqrt(n) by sqrt(n). Thus, for an optimal visualization,
    n should be square (otherwise, it will be padding).

    Parameters
    ----------
    data: :class:`numpy.ndarray`
        Array to visualize
    filename: str, optional
        It contains a path to a filename. If it's None, it display the image.
        It should not be an empty string and should include the extension.
    title: str, optional
        Title of the plot
    normalize: bool, optional
        Whether or not the function normalizes the data
    padding_value: int or float, optional
        Padding value. If normalized is true, the value should be in [0, 1].

    References
    ----------
    https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb#L15
    """

    # normalize data for display (now: value included between 0 and 1)
    if normalize:
        array = (data - data.min()) / (data.max() - data.min())
    else:
        array = data.copy()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(array.shape[0])))
    padding = (((0, n ** 2 - array.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (array.ndim - 3))  # don't pad the last dimension (if there is one)
    array = np.pad(array, padding, mode='constant', constant_values=padding_value)

    # tile the filters into an image
    array = array.reshape((n, n) + array.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, array.ndim + 1)))
    array = array.reshape((n * array.shape[1], n * array.shape[3]) + array.shape[4:])

    plt.figure(figsize=(10, 14))
    plt.imshow(array)
    plt.axis('off')
    plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
