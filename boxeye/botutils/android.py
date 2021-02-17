import io
from PIL import Image
import pyautogui as pyag
import boxeye
from vectormath import Vector2 as Point
import logging


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
    # TODO: make ppadb get raw img, currently gets PNG
    raw = DEVICE.screencap()
    img = Image.open(io.BytesIO(raw))  # RGBA
    img = img.convert(mode)
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


def set_device(client, device):
    """ set device (assign emulator) """
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


logger.info("configuring boxeye for android emulators")
