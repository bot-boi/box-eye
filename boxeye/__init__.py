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
    # NOTE: bgr or rgb?
    img = _grayscale(img)
    img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # img = ImageOps.grayscale(img)
    # img = img.point(lambda p: p > threshold and 255)
    return img


class Point(vectormath.Vector2):
    def __init__(self, x, y):
        x = int(x)  # default type for Vector2 is float
        y = int(y)  # screencoords dont have fractions
        super().__init__(x, y)


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
        """ default preprocessing for pytesseract ocr """
        p1, p2 = self.region
        print(p1.x)
        img = img[p1.x: p2.x, p1.y: p2.y]
        # crop_img = whole_img.crop((p1.x, p1.y, p2.x, p2.y))
        # img = ImageOps.scale(crop_img, self.scale)
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        img = _binarize(img, threshold=self.threshold)
        if self.invert:
            img = cv2.bitwise_not(img)
            # img = ImageOps.invert(img)
        img = cv2.GaussianBlur(img, (2, 2), cv2.BORDER_DEFAULT)
        # img = img.filter(ImageFilter.GaussianBlur(radius=2))
        # img = img.crop(img.getbbox())
        # img = ImageOps.expand(img, border=10, fill='white')
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
        keys = data.keys()  # key to index mapping
        ki_map = {(k, keys.index(k)) for k in keys}  # key -> index map

        out = []
        for ob in zip(*data.values()):  # ob = object
            conf = ob[ki_map["conf"]] * 0.01  # normalize to .0,1.
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
        img = self._tesseract_improve_quality(img)
        data = pytesseract.image_to_data(img, lang='eng', config=self.config,
                                         output_type=Output.DICT)
        raw = self._tesseract_parse_output(data)
        # parse_output could return just the matches
        matches = [i for i in raw if re.search(self.target, i[2])]

        p1, p2 = self.region
        if self.debug:
            raise NotImplementedError
        #     whole_img.show()
        #     img.show()
        #     print(text)
        #     breakpoint()
        return matches


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
        # if self.mode != img.mode:
        #     img = img.convert(self.mode)
        if self.grayscale:
            img = _grayscale(img)

        # UGLY
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


class Region(): # REVIEW
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


    def readboxes(self, scale=10, threshold=200, invert=True,
                  config='', whole_img=None) -> str:
        if whole_img is None:
            whole_img = capture()
        crop_img = self.crop(whole_img)

        # Tesseract Improving Quality
        img = ImageOps.scale(crop_img, scale)
        img = binarize(img, threshold=threshold)
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
            whole_img = capture()
        # Tesseract Improving Quality
        crop_img = self.crop(whole_img)
        img = ImageOps.scale(crop_img, scale)
        img = binarize(img, threshold=threshold)
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
