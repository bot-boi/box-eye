import inspect
import os
import sys

from retry.api import retry_call
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


# conditional retry decorator, lets me add stuff like "FAILSAFE-MODE"
def retry_if(_condition, *args, **kwargs):
    @decorator
    def mydecorator(func, *dargs, **dwargs):  # decorator args
        if not _condition:
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
