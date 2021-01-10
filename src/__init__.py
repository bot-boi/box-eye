import io
import logging
import sched
import time
import traceback
from time import sleep

import pyautogui as pyag
import pytesseract
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from ppadb.client import Client as AdbClient
from pytesseract import Output
from vectormath import Vector2 as Point

from . import util

CLIENT_WIDTH = 960  # always run bots at this res or lower :)
CLIENT_HEIGHT = 540


logging.getLogger("PIL").setLevel(logging.CRITICAL)
logging.basicConfig(format="%(asctime)s-%(pathname)s-"
                    "%(levelname)s> %(message)s",
                    datefmt="%H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger("mylogger")


client = AdbClient(host="127.0.0.1", port=5037)
default_device = client.device("emulator-5554")
client.remote_connect("192.168.1.10", 9999)  # remote emulators
client.remote_connect("192.168.1.11", 9999)
device10 = client.device("192.168.1.10:9999")
device11 = client.device("192.168.1.11:9999")
device137 = client.device("192.168.48.101:5555")
DEVICE = None


def periodic(scheduler, interval, priority, action, actionargs=(),
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


def screencap(mode="RGB"):
    # TODO: make ppadb get raw img, currently gets PNG
    raw = DEVICE.screencap()
    img = Image.open(io.BytesIO(raw))  # RGBA
    img = img.convert(mode)
    return img


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


def tap(point: Point):
    x, y = _inputarg_handler(point)
    DEVICE.shell("input tap {} {}".format(x, y))
    logger.debug("tap @ {} {}".format(x, y))


def swipe(pt1: Point, pt2: Point, ms=1500):
    x1, y1 = _inputarg_handler(pt1)
    x2, y2 = _inputarg_handler(pt2)
    DEVICE.shell("input swipe {} {} {} {} {}".format(x1, y1, x2, y2, ms))


def hold(pt: Point, ms=6000):
    x, y = _inputarg_handler(pt)
    pt = Point(x, y)
    swipe(DEVICE, pt, pt, ms)


# fake device for running tests on stuff without needing a running emulator
class TestDevice:
    def __init__(self, img):
        if img is None:
            self.img = None
            return
        import io
        if isinstance(img, str):
            if ".png" not in img:
                img += ".png"
            img = Image.open(img)
        img = img.convert("RGB")
        bytes = io.BytesIO()
        img.save(bytes, format='PNG')
        self.img = bytes.getvalue()

    def screencap(self):
        return self.img


class RetryError(Exception):
    pass


class NotOpenError(Exception):
    def __init__(self, func):
        import inspect
        self.expression = str(inspect.getsource(func))
        self.message = "Failed to open {}".format(func)


class OCRError(Exception):
    pass


def set_device(device):
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
        logger.info("waiting for {} to load, sleeping 30s..."
                    .format(my_app))
        sleep(30)  # TODO: dont blind wait?


class Pattern():
    def __init__(self, name=None, confidence=0.8, region=None):
        """__init__.

        :param name: the name of the pattern
        :type name: str
        :param confidence: how confident of match u need to be
        :type confidence: float
        :param region: the area to search
        :type region: (Point, Point)

        """
        if name is None:
            raise Exception("unnamed pattern!")
        # if name is None:
        #     name = "NONAME"
        self.name = name
        self.confidence = confidence
        if region is None:
            region = (Point(0, 0), Point(CLIENT_WIDTH - 1, CLIENT_HEIGHT - 1))
        self.region = region

    def isvisible(self, img=None):
        if img is None:
            img = screencap()
        return len(self.locate(img=img)) > 0


# TODO: ignore_case option
class TextPattern(Pattern):
    """TextPattern."""

    def __init__(self, target: str, scale=10,
                 threshold=200, invert=True, config="--psm 8",
                 name=None, debug=False, **kwargs):
        """__init__.

        :param target:
        :type target: str
        :param scale: preproc scaling
        :type scale: int
        :param threshold: binarization threshold
        :type threshold: int
        :param invert: whether to invert or not
        :type invert: bool
        :param config: tesseract config
        :type config: str
        :param kwargs: See Pattern for more args...

        """
        self.target = target
        self.scale = scale
        self.threshold = threshold
        self.invert = invert
        self.config = config
        if name is None:
            name = self.target
        self.name = name
        self.debug = debug
        super().__init__(name=name, **kwargs)

    def __str__(self):
        return "TxtPat-{}".format(self.name)

    def locate(self, img=None):
        whole_img = img
        p1, p2 = self.region
        if whole_img is None:
            whole_img = screencap()

        # Tesseract Improving Quality
        crop_img = whole_img.crop((p1.x, p1.y, p2.x, p2.y))
        img = ImageOps.scale(crop_img, self.scale)
        img = util.binarize(img, threshold=self.threshold)
        if self.invert:
            img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        img = img.crop(img.getbbox())
        # img = ImageOps.expand(img, border=10, fill='white')
        # feed to tesseract, remove newlines (\r \n \x0c etc)
        (_, crop_height) = crop_img.size
        (width, height) = whole_img.size
        draw = ImageDraw.Draw(whole_img)
        data = pytesseract.image_to_data(img, lang='eng', config=self.config,
                                         output_type=Output.DICT)
        out = []
        for text, conf, x1, y1, w, h in zip(data["text"], data["conf"],
                                            data["left"], data["top"],
                                            data["width"], data["height"]):
            conf = float(conf) * 0.01
            text = text.replace(" ", "")
            if text == self.target and \
               conf >= self.confidence and len(text) > 0:

                x2 = x1 + w  # convert tesseract output to top left origin
                y2 = y1 + h
                x1 = x1 // self.scale
                y1 = y1 // self.scale
                x2 = x2 // self.scale
                y2 = y2 // self.scale
                x1 += p1.x  # offset to mainscreen
                y1 += p1.y
                x2 += p1.x
                y2 += p1.y
                p1 = Point(x1, y1)
                p2 = Point(x2, y2)
                out.append((p1, p2))
                draw.rectangle([p1.x, p1.y, p2.x, p2.y],
                               outline="red", width=2)

        if self.debug:
            whole_img.show()
            img.show()
            print(text)
            breakpoint()

        if self.target not in data["text"]:  # UGLY
            return []
        return out


class ImagePattern(Pattern):
    """ImagePattern."""

    def __init__(self, target: Image, grayscale=False,
                 mode="RGB", **kwargs):
        """__init__.

        :param target:  the image to find
        :type target: PIL.Image | str
        :param grayscale: apply grayscaling (faster)
        :type grayscale: bool
        :param mode: PIL Image mode (auto convert)
        :type mode: str
        :param kwargs: See Pattern for more args...

        """
        path = None
        if isinstance(target, str):
            self.path = target
            target = Image.open(target)
        target = target.convert(mode)
        if grayscale:
            target = ImageOps.grayscale(target)

        self.path = path
        self.target = target
        self.grayscale = grayscale
        self.mode = mode
        super().__init__(**kwargs)

    def __str__(self):
        return "ImgPat-{}".format(self.fname)

    def locate(self, img=None):
        if img is None:
            img = screencap()
        if self.mode != img.mode:
            img = img.convert(self.mode)
        if self.grayscale:
            img = ImageOps.grayscale(img)

        # UGLY -- TODO: highlight "UGLY"
        # gotta convert to pyag's region (x0, y0, x1, y1)
        p1, p2 = self.region
        region = (int(p1.x), int(p1.y),
                  int(p2.x), int(p2.y))
        res = pyag.locate(self.target, img,
                          confidence=self.confidence,
                          region=region)
        if "ads" not in self.name:
            logger.debug("located {} at {}".format(self.name, res))
        return [res] if res is not None else []


class PatternList(Pattern):
    def __init__(self, data=[], match_all=False, **kwargs):
        self.data = data
        self.match_all = match_all
        super().__init__(**kwargs)

    def __str__(self):
        return "PList-{}".format(self.name)

    def append(self, p: Pattern):
        self.data.append(p)

    def locate(self, img=None, incl_names=False):
        if img is None:
            img = screencap()
        res = None
        if incl_names:
            # FIXME BUG HERE ?
            res = [(p.name, p.locate()) for p in self.data]
            res = [(n, l) for (n, l) in res if len(l) > 0]  # filter empty
        else:
            res = [p.locate() for p in self.data]
            res = [loc for loc in res if len(loc) > 0]
        return res

    def isvisible(self, img=None):
        bools = [p.isvisible(img=img) for p in self.data]
        if self.match_all:
            return False not in bools
        else:  # match any
            return True in bools


class Region():
    """Region.

    Region is for when you want to read text from
        somewhere on the screen.

    """
    def __init__(self, p1: Point, p2: Point):
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        self.p1 = p1  # closer to 0,0
        self.p2 = p2  # farther from 0,0

    @property
    def width(self):
        return (self.p2 - self.p1).x

    @property
    def height(self):
        return (self.p2 - self.p1).y

    @property
    def middle(self):
        return ((self.p2 - self.p1) // 2) + self.p1

    @property
    def region(self):
        return (self.p1, self.p2)

    # @property
    # def region(self):
    #     return (self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    def tap(self):
        logger.debug("tap @ {}".format(self.middle))
        tap(self.middle)

    def hold(self, ms=3000):
        hold(self.middle, ms=ms)
        logger.debug("holding @ {} for {}ms".format(self.middle, ms))

    def crop(self, img: Image) -> Image:
        return img.crop((self.p1.x, self.p1.y, self.p2.x, self.p2.y))

    def readtext(self, scale=10, threshold=200, invert=True,
                 config="", img=None) -> str:
        if img is None:
            img = screencap()
        img = self.crop(img)

        # Tesseract Improving Quality
        img = ImageOps.scale(img, scale)
        img = util.binarize(img, threshold=threshold)
        if invert:
            img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        img = img.crop(img.getbbox())
        img = ImageOps.expand(img, border=10, fill='white')  # this is imprtant
        # img.show()
        # feed to tesseract, remove newlines (\r \n \x0c etc)
        text = pytesseract.image_to_string(img, lang='eng', config=config)
        text = "".join(text.splitlines())
        logger.debug("read text {}".format(text))
        return text

    def readboxes(self, scale=10, threshold=200, invert=True,
                  config='', whole_img=None) -> str:
        if whole_img is None:
            whole_img = screencap()
        crop_img = self.crop(whole_img)

        # Tesseract Improving Quality
        img = ImageOps.scale(crop_img, scale)
        img = util.binarize(img, threshold=threshold)
        if invert:
            img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        img = img.crop(img.getbbox())
        img = ImageOps.expand(img, border=10, fill='white')  # this is imprtant
        # feed to tesseract, remove newlines (\r \n \x0c etc)
        boxes = pytesseract.image_to_boxes(img, lang='eng', config=config)
        (_, crop_height) = crop_img.size
        (width, height) = whole_img.size
        boxes = [b for b in boxes.splitlines()]
        nboxes = []
        # draw = ImageDraw.Draw(whole_img)
        for box in boxes:
            box = box.split(" ")
            c = box[0]  # the char that was found
            box = [int(i) // scale for i in box[1:]]
            (x1, y2, x2, y1, _) = box  # last value is page number
            # offset to screen coordinates cus of cropping
            ny1 = crop_height - y1  # convert from bot left (tesseract)
            ny2 = crop_height - y2  # to top left origin (swap y and flip)
            x1 += self.p1.x
            ny1 += self.p1.y
            x2 += self.p1.x
            ny2 += self.p1.y
            p1 = Point(x1, ny1)
            p2 = Point(x2, ny2)
            box = (c, (p1, p2))
            # logger.debug(box)
            nboxes.append(box)
            # draw.rectangle([p1.x, p1.y, p2.x, p2.y], outline="red", width=3)

        # whole_img.show()
        # img.show()
        return nboxes

    def readwordboxes(self, scale=10, confidence=0.8, threshold=200,
                      invert=True, config='--psm 11',
                      whole_img=None) -> str:
        if whole_img is None:
            whole_img = screencap()
        # Tesseract Improving Quality
        crop_img = self.crop(whole_img)
        img = ImageOps.scale(crop_img, scale)
        img = util.binarize(img, threshold=threshold)
        if invert:
            img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        # img = img.crop(img.getbbox())
        img = ImageOps.expand(img, border=10, fill='white')  # this is imprtant

        # feed to tesseract, remove newlines (\r \n \x0c etc)
        (_, crop_height) = crop_img.size
        (width, height) = whole_img.size
        # draw = ImageDraw.Draw(whole_img)
        data = pytesseract.image_to_data(img, lang='eng', config=config,
                                         output_type=Output.DICT)
        out = []
        for text, conf, x1, y1, w, h in zip(data["text"], data["conf"],
                                            data["left"], data["top"],
                                            data["width"], data["height"]):
            conf = float(conf)
            text = text.replace(" ", "")
            if conf >= confidence and len(text) > 0:
                x2 = x1 + w
                y2 = y1 + h
                x1 = x1 // scale
                y1 = y1 // scale
                x2 = x2 // scale
                y2 = y2 // scale
                x1 += self.p1.x  # offset to mainscreen
                y1 += self.p1.y
                x2 += self.p1.x
                y2 += self.p1.y
                p1 = Point(x1, y1)
                p2 = Point(x2, y2)
                out.append((text, conf, (p1, p2)))
                # draw.rectangle([p1.x, p1.y, p2.x, p2.y],
                #                outline="red", width=2)

        # whole_img.show()
        # img.show()
        return out

    def readnumbers(self, scale=10, threshold=200, config="--psm 8 -c "
                    "tessedit_char_whitelist=,./0123456789IKMO",
                    img=None):
        """readnumbers.
        do "tesseract --help-extra" for config options
        :param scale:
        :param threshold:
        :param config:

        """
        text = self.readtext(scale, threshold=threshold,
                             config=config, img=img)
        text = text.replace(" ", "")
        text = text.replace(",", "")
        text = text.replace("I", "1")
        text = text.replace("O", "0")
        text = text.replace("/", "7")  # this is pushing it...
        # this way is better cus it can handle abbreviations like 1.5M
        mul = 1
        if "K" in text:
            text = text.replace("K", "")
            mul = 1000
        elif "M" in text:
            text = text.replace("M", "")
            mul = 1000000

        # text = "".join([i for i in text if i.isdigit()])
        n = float(text) * mul
        if text == '':
            n = None
        logger.debug("read number as {}".format(n))
        return n


# TODO: remove this, popslots uses it?
class Task():
    def __init__(self, interval, priority, action, actionargs=(),
                 interval_on_err=None):
        self.interval = interval
        self.priority = priority
        self.action = action
        self.actionargs = actionargs
        if interval_on_err is None:
            interval_on_err = interval // 4
        self.interval_on_err = interval_on_err
        self.time = time.time()

    def __str__(self):
        return "{}-{}-{}".format(self.action.__name__,
                                 self.priority,
                                 self.interval)

    def set_timer(self, ttime):
        self.time = ttime

    def reschedule(self):
        self.time = time.time() + self.interval

    # override this
    def reschedule_on_err(self):
        self.time = time.time() + self.interval_on_err

    def run(self):
        self.action(self.actionargs)
        self.reschedule()


class TaskManager():
    def __init__(self, device):
        self.device = device
        self.queue = []

    def enter(self, task: Task):
        self.queue.append(task)

    def run(self):
        err_counter = 0
        # check if err > err_max every err_interval
        err_interval = 60 * 5
        err_max = 40
        err_next_interval = time.time() + err_interval
        prev_len = len(self.queue)
        while True:
            if len(self.queue) != prev_len:
                prev_len = len(self.queue)
                self.queue.sort(key="time")  # TODO: get this tested
                self.queue.sort(key="priority")
            for i, task in enumerate(self.queue):
                if time.time() >= task.time:
                    try:
                        logger.debug("running task {}".format(task))
                        task.run()
                    except RetryError:  # as e:
                        # reset the task clock for a retry
                        # print("traceback ", traceback.format_exc())
                        task.reschedule_on_err()
                    except Exception as e:
                        traceback.print_exc()
                        if err_next_interval >= time.time():
                            err_counter = 0
                            err_next_interval = time.time() + err_interval
                        if err_counter >= err_max:
                            img = screencap(self.device)
                            img.save('error.png')
                            img.show()
                            raise e
                        else:
                            err_counter += 1


# TODO: FIGURE OUT BETTER CONCEPT FOR EVENT ADDER FUNC  !!!UGLY!!!
def make_main(scheduler, app_pkgname, event_adder_func):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('device', metavar='DEVICE',
                        help='the adb device you want to use')
    parser.add_argument('--game', default='god-of-sky',
                        help='name of the slot game to play')
    parser.add_argument('--mode', action='store',
                        help='select bot mode [DEBUG, NOFAIL]')

    # parse and use args
    args = parser.parse_args()
    if "emulator" not in args.device:  # remote device
        ip, port = args.device.split(":")
        client.remote_connect(ip, int(port))
    device = client.device(args.device)
    game = args.game
    mode = args.mode


    return main, game


class BotRunner:
    def __init__(self, app_pkgname):
        self.scheduler = sched.scheduler()
        self.app_pkgname = app_pkgname

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('device', metavar='DEVICE',
                            help='the adb device you want to use')
        parser.add_argument('--game', default='god-of-sky',
                            help='name of the slot game to play')
        parser.add_argument('--mode', action='store',
                            help='select bot mode [DEBUG, NOFAIL]')

        # parse and use args
        args = parser.parse_args()
        if "emulator" not in args.device:  # remote device
            ip, port = args.device.split(":")
            client.remote_connect(ip, int(port))
        self.device = client.device(args.device)
        self.game = args.game
        self.mode = args.mode
        if self.mode == 'DEBUG':
            import core.debug_mode  # run pdb on failure

        set_device(self.device)
        launch_app(self.app_pkgname)

    def enter_periodic(self, *args, **kwargs):
        periodic(self.scheduler, *args, **kwargs)
        logger.debug("added periodic event to BotRunner {}"
                     .format(self.app_pkgname))

    def run(self):
        if len(self.scheduler._queue) == 0:
            raise Exception("BotRunner event queue empty on start!")
        scheduler.run()


# =============== what use to be slots.py ==================
# only here in case of backwards compatibility problems
# TODO: remove this
class ImageRegion(Region):
    def __init__(self, paths, p1: Point, p2: Point, device,
                 grayscale=False, match_all=True,
                 confidence=None):
        imgs = None
        if not isinstance(paths, list):
            paths = [paths]
        if len(paths) == 0:
            raise Exception("paths is none")

        if isinstance(paths[0], str):
            paths = list(map(lambda i: i + ".png" if ".png" not in i else i,
                             paths))
            imgs = [Image.open(p) for p in paths]

        if grayscale:
            imgs = [ImageOps.grayscale(i) for i in imgs]
        else:
            imgs = [i.convert("RGB") for i in imgs]

        self.imgs = imgs
        self.grayscale = grayscale
        self.match_all = match_all
        self.paths = paths
        super().__init__(p1, p2, device)
        # REVIEW: do this different
        self.confidence = confidence

    def isvisible(self, confidence=0.95) -> bool:
        if self.confidence is not None:
            confidence = self.confidence
        img = screencap(self.device)
        if self.grayscale:
            img = ImageOps.grayscale(img)

        res = [pyag.locate(i, img, confidence=confidence,
                           region=self.region)
               for i in self.imgs]
        res = [i for i in res if i is not None]
        # print(list(self.paths), len(res), len(self.imgs))
        if self.match_all:
            return (len(res) == len(self.imgs))  # match all
        return (len(res) > 0)  # match any


class TextRegion(Region):
    def __init__(self, text, p1, p2, device,
                 match_all=True, invert=True,
                 threshold=200, config=""):
        if type(text) is str:
            text = [text]
        self.text = text
        self.match_all = match_all
        self.invert = invert
        self.threshold = threshold
        self.config = config
        super().__init__(p1, p2, device)

    def isvisible(self, **kwargs):
        read = self.readtext(invert=self.invert,
                             threshold=self.threshold,
                             config=self.config)
        result = map(lambda s: s in read, self.text)
        if self.match_all:
            return (False not in result)
        else:
            return (True in result)


class Interface:
    def __init__(self, device):
        self.device = device

    def open(self) -> bool:  # try to open, return bool
        pass


# class TestRegion(unittest.TestCase):
#     def setUp(self):
#         self.device = TestDevice(APP_PATH + "test/surprise-gift.png")
#         self.region = Region(Point(0, 0), Point(CLIENT_WIDTH - 1,
#                                                 CLIENT_HEIGHT - 1),
#                              self.device)
# 
#     def test_readboxwords(self):
#         boxes = self.region.readwordboxes()
#         logger.debug("readwordboxes {}".format(boxes))
#         self.assertTrue(len(boxes) > 0)