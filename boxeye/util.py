import cv2 as cv


def grayscale(img):
    """ grayscale an image with opencv """
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def binarize(img, threshold=150):
    """ 0,1 an image by converting to grayscale and then thresholding.
    Uses a single threshold value.

    Parameters
    ----------
    img : np.ndarray
        the image to binarize
    threshold : int
        img is split into 0,1 along this value (0-255)

    Returns
    -------
        img : np.ndarray
            the binarized image
    """
    img = grayscale(img)
    # what is the first value returned here for? \/
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img
