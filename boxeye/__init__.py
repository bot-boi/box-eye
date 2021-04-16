import logging
import os
import re

import cv2 as cv
import numpy as np
import pytesseract
import vectormath

from pytesseract import Output

from .botutils import android
from .cts import CTS2


try:
    MAXDEBUG = os.environ["MAXDEBUG"]
    MAXDEBUG = True
except KeyError:
    MAXDEBUG = False


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("boxeye")
logger.setLevel(logging.DEBUG)


class NoDepsException(Exception):
    def __init__(self):
        super().__init__("You need to instantiate boxeye with a dependency.")


# TODO: require certain parameters to be present in injected deps
#       for capture, click, and drag ?  how to do that ?
def capture(*args, **kwargs):
    return android.capture(*args, **kwargs)


def click(*args, **kwargs):
    return android.click(*args, **kwargs)


# drag should also hold (drag from A to A)
def drag(*args, **kwargs):
    return android.drag(*args, **kwargs)


def keypress(*args, **kwargs):
    return android.keypress(*args, **kwargs)

# def inject(depname):
#     """ inject dependencies (capture, click, drag) """
#     global capture
#     global click
#     global drag
#     if depname == "android":
#         # this will have to do for now...
#         from boxeye.botutils import android
#         capture = android.capture
#         click = android.click
#         drag = android.drag


def _grayscale(img):
    """ grayscale an image with cv """
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


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
    img = _grayscale(img)
    # what is the first value returned here for? \/
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img


def Point(x, y):  # enforce int
    return vectormath.Vector2(x, y).astype(int)


class Pattern():
    def __init__(self, name=None, confidence=0.95, region=None, debug=False):
        """
            name : str
                the name of the pattern (for log)
            confidence : float = 0.95
                the minimum confidence required to match
            region : (boxeye.Point, boxeye.Point)
                the area of the screen to search in

        """
        if name is None:
            raise Exception("unnamed pattern!")
        self.name = name
        self.confidence = confidence
        self.region = region
        self.debug = debug

    def isvisible(self, img=None):
        return len(self.locate(img=img)) > 0


# TODO: ignore_case option?
#       ignore case by default?
class TextPattern(Pattern):
    """TextPattern."""
    def __init__(self, target: str, scale=10,
                 threshold=200, invert=True, config="--psm 8",
                 name=None, **kwargs):
        """__init__.

        :param target:
        :type target: regex
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
        super().__init__(name=name, **kwargs)

    def __str__(self):
        return "TxtPat-{}".format(self.name)

    def _tesseract_improve_quality(self, img):
        """
            default preprocessing for pytesseract ocr
            see the "Tesseract Improving Image Quality" page online
        """
        p1, p2 = self.region
        img = img[p1.y: p2.y, p1.x: p2.x]  # NOTE: np.array is col,row
        img = cv.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        img = _binarize(img, threshold=self.threshold)
        if self.invert:
            img = cv.bitwise_not(img)
        img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
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

    def locate_names(self, img=None):
        if img is None:
            img = capture()
        whole_img = img
        img = self._tesseract_improve_quality(img)
        data = pytesseract.image_to_data(img, lang='eng', config=self.config,
                                         output_type=Output.DICT)
        raw = self._tesseract_parse_output(data)
        # raw format is ((p1, p2), text, confidence)
        # parse_output could return just the matches
        matches = []
        for i in raw:
            if re.search(self.target, i[1]):
                if i[2] >= self.confidence:
                    matches.append(i)
                else:
                    logger.debug("failed to match {}, {.3f} < {}"
                                 .format(self.name, i[2], self.confidence))

        if self.debug or MAXDEBUG:
            cv.namedWindow("debug", flags=cv.WINDOW_GUI_NORMAL)
            cv.moveWindow("debug", 0, 0)
            cv.imshow("debug", whole_img)
            cv.waitKey(6000)
            cv.imshow("debug", img)
            cv.waitKey(6000)
            breakpoint()

        # logger.debug("got {} matches for {}".format(len(matches), self.name))
        return matches

    def locate(self, img=None):
        """ return just regions """
        return [i[0] for i in self.locate_names(img=img)]


class NumberReader(TextPattern):
    """ reads numbers from an area """
    def __init__(self, target, config="--oem 0 --psm 8 -c "
                 "tessedit_char_whitelist=,.0123456789KM", **kwargs):
        super().__init__(target, config=config, **kwargs)

    def get(self, img=None):
        raw = self.locate(img=img)
        if len(raw) <= 0:
            return None
        raw_strings = [i[1] for i in raw]
        text = raw_strings[0]  # TODO: pick match with highest confidence?
        if self.debug:
            breakpoint()
        if "K" in text:
            text = text.replace("K", "")
            return float(text) * 1000.0
        elif "M" in text:
            text = text.replace("M", "")
            return float(text) * 1000000.0
        else:
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
            target = cv.imread(target, cv.IMREAD_COLOR)
            # it probably isnt necessary to convert to RGB
            # the output of *capture* will need to be BGR
            # target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
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
        whole_img = img
        if whole_img is None:
            whole_img = capture()
        if self.grayscale:
            # self.target is already grayscaled
            whole_img = _grayscale(whole_img)

        # gotta convert to pyag's region (x0, y0, x1, y1)
        p1, p2 = self.region
        img = whole_img[p1.y: p2.y, p1.x: p2.x]  # crop to this patterns region
        res = cv.matchTemplate(img, self.target, cv.TM_CCORR_NORMED)
        # res = cv.normalize(res, res, 0, 1, cv.NORM_MINMAX, -1)
        # loc = np.where(res >= self.confidence)
        # # convert raw to point and reapply offset (p1)
        # points = [(Point(pt[0], pt[1]) + p1) for pt in zip(*loc[::-1])]
        if self.grayscale:
            h, w = self.target.shape
        else:
            h, w, _ = self.target.shape
        # matches = [(pt, pt + Point(pt.x + w, pt.y + h)) for pt in points]
        _, max_val, _, mloc = cv.minMaxLoc(res)
        mloc += p1  # reapply offset cus we cropped earlier
        matched = []
        if max_val >= self.confidence:
            matched.append((Point(mloc[0], mloc[1]),
                            Point(mloc[0] + w, mloc[1] + h)))
        else:
            logger.debug("failed to match {}, {:.2f} < {}"
                         .format(self.name, max_val, self.confidence))

        if self.debug or MAXDEBUG:
            drawn_img = whole_img
            for p1, p2 in matched:
                drawn_img = cv.rectangle(whole_img, tuple(p1), tuple(p2),
                                         (255, 0, 0), 2)
            cv.namedWindow("debug", flags=cv.WINDOW_GUI_NORMAL)
            cv.moveWindow("debug", 0, 0)
            cv.imshow("debug", drawn_img)
            cv.waitKey(6000)
            breakpoint()

        # logger.debug("located {} at {}".format(self.name, matched))
        return matched


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


class ColorPattern(Pattern):
    def __init__(self, cts, cluster=5, min_thresh=50, max_thresh=5000,
                 **kwargs):
        """
            cts : CTS
                - the *color tolerance speed (aka method)*
            cluster : int = 5 (pixels)
                - the radius to use when clustering
        """
        self.cluster = cluster
        if isinstance(cts, list):
            cts = CTS2.fromarray(cts)
        self.cts = cts
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        super().__init__(**kwargs)

    def locate_points(self, img=None):
        """ Find all points in an image that match a cts.
            Then group them with radius, apply thresholds
            and maybe some other stuff.
        """
        (h, w, _) = img.shape  # REVIEW ?
        if self.region is None:
            region = (Point(0, 0), Point(w, h))
        else:
            region = self.region

        p1, p2 = region
        img = img[p1.y:p2.y, p1.x:p2.x]  # apply bounds
        y, x = np.where(np.logical_and(np.all(img <= self.cts.max, 2),
                                       np.all(img >= self.cts.min, 2)))
        points = np.column_stack((x, y))  # x,y order for PointArrays
        if len(points) <= 0:
            logging.debug("failed to find {}".format(self.name))
            return

        # clustering
        clusters = np.zeros(len(points), dtype='uint32')
        while True:  # loop until all points are clustered
            unclustered = clusters == 0
            remaining = np.count_nonzero(unclustered)
            if remaining == 0:
                break
            candidate = points[unclustered][np.random.randint(remaining)]
            dist = np.sum(np.square(points - candidate), axis=1)
            nearby_mask = dist <= self.cluster * self.cluster
            overlaps = set(list(clusters[nearby_mask]))
            overlaps.remove(0)
            if len(overlaps) == 0:
                G = np.max(clusters)+1  # new cluster
            else:
                G = np.min(list(overlaps))  # prefer smaller numbers
            clusters[nearby_mask] = G
            for g in overlaps:
                if g == G or g == 0:
                    continue
                clusters[clusters == g] = G
        unique, counts = np.unique(clusters, return_counts=True)
        clustered_points = np.array([points[clusters == c] for c in unique])
        filtered_points = [clus for clus in clustered_points
                           if len(clus) >= self.min_thresh
                           and len(clus) <= self.max_thresh]
        return filtered_points

    def locate(self, img=None):
        points = self.locate_points(img=img)
        # REVIEW: does this work ?  points sorted or no ?
        regions = [(Point(cl[0][0], cl[0][1]),
                    Point(cl[-1][0], cl[-1][1]))
                   for cl in points]
        logging.debug("found @ {}".format(regions))
        return regions
