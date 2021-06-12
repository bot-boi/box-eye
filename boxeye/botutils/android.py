import logging

import cv2 as cv
import numpy as np
import pyautogui as pyag

from ppadb.client import Client  # adb client
from vectormath import Vector2 as Point

logger = logging.getLogger("boxeye")
DEVICE = None


def _inputarg_handler(point) -> (int, int):
    x = None
    y = None

    # case: Box(l, t, w, h) -- pyautogui
    if isinstance(point, pyag.pyscreeze.Box):
        x1, y1, w, h = point
        x = x1 + (w // 2)
        y = y1 + (h // 2)
    elif isinstance(point, tuple) and isinstance(list(point)[0], Point):
        p1, p2 = point  # case: (Point, Point) aka region
        dist = p2 - p1
        idk = p1 + (dist // 2)
        x = idk.x
        y = idk.y
    elif isinstance(point, Point) or isinstance(point, tuple):
        x, y = point
    else:
        breakpoint()
        raise Exception("bad input args")
    return (x, y)


def capture(mode="RGB"):
    # def decode_raw(bytelist):
    #     raw = np.array(bytelist)
    #     r = raw[0::3]  # get every 3rd
    #     b = raw[1::3]  # ^ offset by 1
    #     g = raw[2::3]
    #     return np.dstack((r, g, b))
    #
    # TODO: get png=False decoder working
    #       and check if its faster

    raw = DEVICE.screencap()
    img = cv.imdecode(np.array(raw), cv.IMREAD_COLOR)
    return img


def drag(pt1: Point, pt2: Point, ms=1500):
    x1, y1 = _inputarg_handler(pt1)
    x2, y2 = _inputarg_handler(pt2)
    DEVICE.shell("input swipe {} {} {} {} {}".format(x1, y1, x2, y2, ms))
    logger.debug("swipe from {},{} to {},{} in {}ms"
                 .format(x1, y1, x2, y2, ms))


def _hold(point: Point, duration=6000):
    x, y = _inputarg_handler(point)
    point = Point(x, y)
    drag(point, point, duration)


def click(point: Point, duration=None):
    if duration is not None:
        _hold(point, duration)
        return
    x, y = _inputarg_handler(point)
    DEVICE.shell("input tap {} {}".format(x, y))
    logger.debug("tap @ {} {}".format(x, y))


def keypress(keycode):
    DEVICE.shell("input keyevent {}".format(keycode))


def set_device(client, device):
    """ set device (assign emulator)
        pass none for default client
    """
    if client is None:
        client = Client()
    if "emulator" not in device:
        ip, port = device.split(":")
        client.remote_connect(ip, int(port))
    device = client.device(device)
    global DEVICE
    DEVICE = device


def current_activity() -> str:
    """
    get the name of the current (primary?) activity and package

    ...
    :returns: str in form "package/activity"
    """
    return DEVICE.shell("dumpsys activity | grep -E mResumedActivity")


def launch_app(my_app: str):
    # i.e. com.huuuge.casino.slots
    if my_app not in current_activity():
        logger.info("launching app {}".format(my_app))
        DEVICE.shell("monkey -p {} 1".format(my_app))


logging.info("running in mode: android")


def testcap():
    from ppadb.client import Client
    client = Client("127.0.0.1", 5037)
    set_device(client, "192.168.1.10:9999")
    return capture()
