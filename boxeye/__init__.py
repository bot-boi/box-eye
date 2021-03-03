import cv2
import pyautogui as pyag
import pytesseract
import re
import vectormath
from pytesseract import Output
import logging
from . import botutils


logger = logging.getLogger("boxeye")


# TODO: require dependency injection on import
#       like `import boxeye; boxeye = boxeye(dep_inj);`
def capture(*args, **kwargs):
    return botutils.android.capture(*args, **kwargs)


def click(*args, **kwargs):
    return botutils.android.click(*args, **kwargs)


def drag(*args, **kwargs):
    return botutils.android.drag(*args, **kwargs)


def _grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _binarize(img, threshold=150):
    """
    .. _binarize:

    0,1 an image by converting to grayscale and then thresholding.
    Uses a single threshold value.

    :param img: the image to binarize.
    :type img: np.array
    :param threshold: img is split into 0,1 along this value (0-255)
    :type threshold: float
    :returns: binarized image
    :rtype: np.array

    """
    # NOTE: bgr or rgb?  does it matter?
    img = _grayscale(img)
    # what is the first value returned here for? \/
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img


def Point(x, y):  # enforce int
    return vectormath.Vector2(x, y).astype(int)


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
        self.name = name
        self.confidence = confidence
        self.region = region

    def isvisible(self, img=None):
        return len(self.locate(img=img)) > 0


# TODO: ignore_case option?
#       ignore case by default?
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

    def _tesseract_improve_quality(self, img):
        """
            default preprocessing for pytesseract ocr
            see the "Tesseract Improving Image Quality" page online
        """
        p1, p2 = self.region
        img = img[p1.y: p2.y, p1.x: p2.x]  # NOTE: np.array is y,x
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        img = _binarize(img, threshold=self.threshold)
        if self.invert:
            img = cv2.bitwise_not(img)
        img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        # maybe crop to bounding box around text?
        # and expand borders?
        return img

    def _tesseract_transform(self, x1, y1, w, h):
        """
            transform a tesseract position result from tesseract space
            (bot left) to screen space (top left origin)
        """

        # rotate so origin is top left
        x2 = x1 + w
        y2 = y1 + h
        x1 = x1 // self.scale
        y1 = y1 // self.scale
        x2 = x2 // self.scale
        y2 = y2 // self.scale

        # offset to mainscreen
        mp1, mp2 = self.region  # my point
        xoff = mp1.x
        yoff = mp1.y
        x1 += xoff
        y1 += yoff
        x2 += xoff
        y2 += yoff
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        return (p1, p2)

    def _tesseract_parse_output(self, data):
        """ parse output from pytesseract """
        logger.debug("parsing tesseract output")
        # keys = data.keys()  # key to index mapping
        ki_map = {k: i for i, k in enumerate(data.keys())}  # key -> index map

        out = []
        for ob in zip(*data.values()):  # ob = object
            conf = float(ob[ki_map["conf"]]) * 0.01  # normalize to 0,1
            text = ob[ki_map["text"]]
            w = ob[ki_map["width"]]
            h = ob[ki_map["height"]]
            x1 = ob[ki_map["left"]]
            y1 = ob[ki_map["top"]]
            # p1, p2 = self._tesseract_transform(x1, y1, w, h)
            region = self._tesseract_transform(x1, y1, w, h)
            out.append((region, text, conf))
        return out

    def locate(self, img=None):
        if img is None:
            img = capture()
        whole_img = img
        img = self._tesseract_improve_quality(img)
        data = pytesseract.image_to_data(img, lang='eng', config=self.config,
                                         output_type=Output.DICT)
        raw = self._tesseract_parse_output(data)
        # raw format is ((p1, p2), text, confidence)
        # parse_output could return just the matches
        matches = [i for i in raw if re.search(self.target, i[1])
                   and i[2] >= self.confidence]

        if self.debug:
            cv2.namedWindow("debug", flags=cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow("debug", 0, 0)
            cv2.imshow("debug", whole_img)
            cv2.waitKey(6000)
            cv2.imshow("debug", img)
            cv2.waitKey(6000)
        return matches


class NumberReader(TextPattern):
    """ reads numbers from an area """
    def __init__(self, target, config="--oem 0 --psm 8 -c "
                 "tessedit_char_whitelist=,.0123456789KM", **kwargs):
        super().__init__(target, config=config, **kwargs)

    def get(self, img=None):
        raw = self.locate(img=img)
        raw_strings = [i[1] for i in raw]
        text = raw_strings[0]  # TODO: pick match with highest conf?
        text = text.replace(" ", "")  # this stuff is copied from old Region
        text = text.replace(",", "")  # is it really necessary?
        text = text.replace("I", "1")

        text = text.replace("K", "000")  # * 1000
        text = text.replace("M", "000000")  # * 1000000
        return float(text)


class ImagePattern(Pattern):
    """ImagePattern."""

    def __init__(self, target, grayscale=False,
                 mode="RGB", **kwargs):
        """__init__.

        :param target:  the image to find
        :type target: np.ndarray | str
        :param grayscale: apply grayscaling (faster)
        :type grayscale: bool
        :param mode: PIL Image mode (auto convert)
        :type mode: str
        :param kwargs: See Pattern for more args...

        """
        self.path = None
        if isinstance(target, str):
            self.path = target
            target = cv2.imread(target, cv2.IMREAD_COLOR)
            # it probably isnt necessary to convert to RGB
            # the output of *capture* will need to be BGR
            # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # target = target.convert(mode)
        if grayscale:
            target = _grayscale(target)
            # target = ImageOps.grayscale(target)

        self.target = target
        self.grayscale = grayscale
        # self.mode = mode
        super().__init__(**kwargs)

    def __str__(self):
        return "IPat::{}".format(self.fname)

    def locate(self, img=None):
        if img is None:
            img = capture()
        if self.grayscale:
            img = _grayscale(img)

        # gotta convert to pyag's region (x0, y0, x1, y1)
        p1, p2 = self.region
        region = (p1.x, p1.y, p2.x, p2.y)
        # REVIEW \/ ^ do i still need to convert?
        res = pyag.locate(self.target, img,
                          confidence=self.confidence,
                          region=region)
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

    def locate_names(self, img=None):
        if img is None:
            img = capture()
        names = []
        loc = []
        for pattern in self.data:
            p_loc = pattern.locate()
            loc = loc + p_loc
            p_name = pattern.name
            p_names = len(p_loc) * [p_name]
            names = names + p_names
        return (names, loc)

    def locate(self, img=None):
        return self.locate_names(img=img)[1]

    def isvisible(self, img=None):
        bools = [p.isvisible(img=img) for p in self.data]
        if self.match_all:
            return False not in bools
        else:  # match any
            return True in bools
