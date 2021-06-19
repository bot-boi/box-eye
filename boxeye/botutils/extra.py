import cv2
import inspect
import os
import sys

from vectormath import Vector2 as P  # Point


# ============ ERROR DEFINITIONS ============
class RetryError(Exception):
    pass


class NotOpenError(Exception):
    def __init__(self, func):
        self.expression = str(inspect.getsource(func))
        self.message = "Failed to open {}".format(func)


class OCRError(Exception):
    pass


# ================== utils ===================
def make_check_vision(imgpath):
    """ Make a *check_vision* function for use with tests """
    def check_vision(obj, method_name, *fnargs, expected=True, methodargs=(),
                     **kwargs):
        """ Check some computer vision action that returns a result.
              obj.method_name should accept a keyword argument *img*,
              else it won't work.

            Parameters
            ----------
                obj - the object to test
                method_name - name of the method of the object to test
                fnargs - list of images to run on
                expected - the value that obj.method_name should return
                methodargs - positional args for obj.method_name
                kwargs - keyword args for obj.method_name
        """

        imgs = list(fnargs)
        # process image names
        imgs = [imgpath + img_name for img_name in imgs]
        imgs = [img_name + ".png" for img_name in imgs]
        imgs = [(cv2.imread(img_name, cv2.IMREAD_COLOR), img_name)
                for img_name in imgs]
        method = getattr(obj, method_name)
        for img, img_name in imgs:
            if not (method(img=img, *methodargs, **kwargs) == expected):
                pytest.fail("{} failed on {}".format(method_name, img_name))
        obj.debug = False

    return check_vision


def periodic(scheduler, interval, priority, action, actionargs=[],
             do_now=True):
    """periodic.
    runs a function periodically using sched module

    :param scheduler:
    :param interval:
    :param priority:
    :param action:
    :param actionargs:
    :param do_now: whether to run at very start or wait for interval

    """

    event = scheduler.enter(interval, priority, periodic,
                            (scheduler, interval, priority,
                             action, actionargs))
    if do_now:
        action(*actionargs)
    return event


# flattens a 2d array
flatten = lambda t: [item for sublist in t for item in sublist
                     if not isinstance(item, tuple)]


def get_center(p1, p2):
    """
    Expects vectormath vector2 type (aka Point)
    """
    if isinstance(p1, tuple):  # form (x,y,w,h)
        (x, y, w, h) = p1
        p1 = P(x, y)
        p2 = P(x + w, y + h)

    return ((p2 - p1) // 2) + p1


def get_app_path(appdir=__file__):
    """ get the application path even when pyinstaller bundle """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(appdir))
