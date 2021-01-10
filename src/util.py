import inspect
import os
import os.path
import re
import sys

from PIL import Image, ImageOps
from vectormath import Vector2 as Point
import pytest
from retry.api import retry_call
from decorator import decorator


# conditional retry decorator, lets me add stuff like "FAILSAFE-MODE"
def retry_if(condition, *args, **kwargs):
    @decorator
    def mydecorator(func, *dargs, **dwargs):  # decorator args
        if not condition:
            return func
        return retry_call(func, fargs=dargs, fkwargs=dwargs,
                          *args, **kwargs)
    return mydecorator


# flattens a 2d array
flatten = lambda t: [item for sublist in t for item in sublist
                     if not isinstance(item, tuple)]


def get_center(p1, p2):
    """
    Expects vectormath vector2 type (aka Point)
    """
    if isinstance(p1, tuple):  # form (x,y,w,h)
        (x, y, w, h) = p1
        p1 = Point(x, y)
        p2 = Point(x + w, y + h)

    return ((p2 - p1) // 2) + p1


# 0,1 an image, threshold is 0-255
def binarize(img, threshold=150):
    """
    .. _binarize:

    0,1 an image by converting to grayscale and then thresholding.
    Uses a single threshold value.

    :param img: the image to binarize.
    :type img: PIL.Image
    :param threshold: img is split into 0,1 along this value (0-255)
    :type threshold: int
    :returns: binarized image
    :rtype: PIL.Image

    """
    img = ImageOps.grayscale(img)
    img = img.point(lambda p: p > threshold and 255)
    return img


def strip_ansi_codes(s: str):  # unused...
    """
    .. _strip_ansi_codes:

    Removes all control sequences (ANSI escape sequences) from a string.
    This is unused because str.splitlines() already does what I want.

    :param s: the string to strip
    :type s: str
    :returns: a string stripped of all fancy ANSI codes
    :rtype: str
    """
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def get_app_path(appdir=__file__):
    """ get the application path even when pyinstaller bundle """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(appdir))


# +++++++++++++++++ TEST UTILS +++++++++++++++++++
def make_check_vision(path):
    def check_vision(pattern, expect_true, expect_false=[]):
        # TODO: add ability to specifiy which method of pattern to use
        #   for assertions
        # handle list and single
        if isinstance(expect_true, str):
            expect_true = [expect_true]
        if isinstance(expect_false, str):
            expect_false = [expect_false]

        # add .png extension if not present and convert to Image
        expect_true = list(map(lambda i: i + ".png" if ".png" not in i else i,
                               expect_true))
        expect_false = list(map(lambda i: i + ".png" if ".png" not in i else i,
                                expect_false))

        # UGLY
        # do the tests
        for fname in expect_true:
            img = Image.open(path + fname)
            res = pattern.isvisible(img=img)
            if res is not True:
                pytest.fail("uh oh {} is not visible in {}"
                            .format(pattern.name, fname))
            assert res is True
        for fname in expect_false:
            img = Image.open(path + fname)
            res = pattern.isvisible(img=img)
            if res is not False:
                pytest.fail("uh oh {} is visible in {}"
                            .format(pattern.name, fname))
            assert res is False

    return check_vision
