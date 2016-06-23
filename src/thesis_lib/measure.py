from skimage.measure import moments, moments_central, moments_normalized, moments_hu


def get_hu_moment_from_image(image):
    """
    Compute the 7 Hu's moments from an image.
    This set of moments is proofed to be translation, scale and rotation invariant.

    Parameters
    ----------
    image: a 2d array of double or uint8

    Returns
    -------
    (7, 1) array of double

    References
    ----------
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.moments
    """
    order = 7
    raw_moments = moments(image, order=order)
    cr = raw_moments[0, 1] / raw_moments[0, 0]
    cc = raw_moments[1, 0] / raw_moments[0, 0]
    central_moments = moments_central(image, cr, cc, order=order)
    normalized_moments = moments_normalized(central_moments, order)
    hu_moments = moments_hu(normalized_moments)
    return hu_moments
