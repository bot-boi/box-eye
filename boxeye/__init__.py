import logging
import os
import re

import cv2 as cv
import numpy as np
import pytesseract
import vectormath

from pytesseract import Output

# TODO: decouple boxeye and botutils
from .botutils import android
from .cts import CTS2
from .point import Point


try:
    MAXDEBUG = os.environ["MAXDEBUG"]
    MAXDEBUG = True
except KeyError:
    MAXDEBUG = False


# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("boxeye")
logger.setLevel(logging.INFO)


class NoDepsException(Exception):
    def __init__(self):
        super().__init__("You need to instantiate boxeye with a dependency.")


# TODO: require certain parameters to be present in injected deps
#       for capture, click, and drag ?  how to do that ?
def capture(*args, **kwargs):
    """ aka screencap """
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
    """ grayscale an image with opencv """
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def _binarize(img, threshold=150):
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
    img = _grayscale(img)
    # what is the first value returned here for? \/
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img


class Pattern():
    """ The Pattern base type.
    A pattern will always have the following:
        * name
        * confidence
        * region
        * debug
        * pause_on_debug
        * isvisible
        * locate
    """
    def __init__(self, name=None, confidence=0.95, region=None, debug=False,
                 pause_on_debug=False):
        """ Create a Pattern.  Meant to be called via super().

        Parameters
        ----------
        name : str
            the name of the pattern (for log)
        confidence : float = 0.95
            the minimum confidence required to match
        region : (boxeye.Point, boxeye.Point)
            the area of the screen to search in
        debug : bool
            true if this pattern is in debug mode
        """
        if name is None:
            raise Exception("unnamed pattern!")
        self.name = name
        self.confidence = confidence
        self.region = region
        self.debug = debug
        self.pause_on_debug = pause_on_debug

    def isvisible(self, img=None):
        """ check if pattern is visible """
        return len(self.locate(img=img)) > 0


# TODO: ignore_case option?
#       ignore case by default?
class TextPattern(Pattern):
    """ Search for text, match against regex """
    def __init__(self, target: str, scale=10,
                 threshold=200, invert=True, config="--psm 8",
                 name=None, **kwargs):
        """ Initialize a TextPattern

        Parameters
        ----------
        target : regex
            the regular expression to match against
        scale : int
            scale text during image preprocessing (5-10)
        threshold : int
            binarization threshold during image preprocessing
        invert : bool
            invert image during preprocessing (after binarization)
        config : str
            tesseract configuration
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
        """ Default preprocessing for pytesseract ocr, see the "Tesseract
            Improving Image Quality" page online for more info.

            This is where cropping, scaling, binarization, and inversion
            are applied.  Also does some default Gaussian blur.
        """
        p1, p2 = self.region
        img = img[p1.y: p2.y, p1.x: p2.x]
        img = cv.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        img = _binarize(img, threshold=self.threshold)
        if self.invert:
            img = cv.bitwise_not(img)
        img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
        return img

    def _tesseract_transform(self, x1, y1, w, h):
        """ Transform a tesseract position result from tesseract space
            (bot left) to screen space (top left origin).

            Parameters
            ----------
            x1, y1: int
                the top left corner of the region we want to rotate
            w, h: int
                the width and height of the region

            Returns
            -------
            region : (Point, Point)
                the newly transformed region
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
        """ Handle raw output from tesseract.

            Basically just transforms points and outputs
            them plus text and confidence.

            Parameters
            ----------
            data : Dict
                the raw output from pytesseract.data_to_data (?)
        """
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
        """ Special function that locates names and confidence in addition
            to the usual region.

            Parameters
            ----------
            img : np.ndarray
                optional image to use for locating

            Returns
            -------
            matches : [((Point, Point), text, confidence)]
                successful match results!
        """
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

        logger.debug('search for {} at {} got the following:\n'
                     '{}'.format(self.name, self.region, matches))
        if self.debug or MAXDEBUG:
            cv.namedWindow("debug", flags=cv.WINDOW_GUI_NORMAL)
            cv.moveWindow("debug", 0, 0)
            cv.imshow("debug", whole_img)
            cv.waitKey(6000)
            cv.imshow("debug", img)
            cv.waitKey(6000)
            if self.pause_on_debug:
                breakpoint()

        return matches

    def locate(self, img=None):
        """ Wrapper for locate_names, returns only the region.

            Parameters
            ----------
            img : np.ndarray
                optional image to use for locating

            Returns
            -------
            matches : [(Point, Point)]
                regions of successful matches!
        """
        matches = [i[0] for i in self.locate_names(img=img)]
        if len(matches) > 0:
            logger.info('Got {} matches for {}.'
                        .format(len(matches), self.name))
        return matches


class NumberReader(TextPattern):
    """ Exactly the same as TextPattern but only reads numbers.
        This is due to the tesseract *config* property.
    """
    def __init__(self, target, config="--oem 0 --psm 8 -c "
                 "tessedit_char_whitelist=,.0123456789KM", **kwargs):
        super().__init__(target, config=config, **kwargs)

    # REVIEW: this should probably be called "locate"
    def get(self, img=None):
        """ Locate and convert found text to numbers.
        """
        raw = self.locate_names(img=img)  # [((p1, p2), text)]
        if len(raw) <= 0:
            return None
        raw_strings = [i[1] for i in raw]
        text = raw_strings[0]  # TODO: pick match with highest confidence?
        # cleanup
        text = text.strip()
        res = None
        if self.debug:
            if self.pause_on_debug:
                breakpoint()
        if text.find("K") >= 0:  # 100K
            text = text.replace("K", "")
            res = float(text) * 1000.0
        elif text.find("M") >= 0:
            text = text.replace("M", "")  # 1M
            res = float(text) * 1000000.0
        else:
            res = float(text)
        return(round(res))


class ImagePattern(Pattern):
    """ Search for an image. """
    def __init__(self, target, grayscale=False, mask=None, multi=False,
                 method=cv.TM_CCORR_NORMED, **kwargs):
        """ Initialize a new ImagePattern.

            Parameters
            ----------
            target : str | np.ndarray
                the target image or image path (aka needle)
            grayscale : bool
                locate using grayscale
            mode : str
                image mode, unused (always RGB)
            mask : np.ndarray (UINT8x1, F32x3)
                apply a mask to the target image using binary/grayscale img
                if target is str and target + "-mask.png" exists in the same
                dir as target then target + "-mask.png" will be loaded as mask
                if target is ndarray then a mask is required
            multi : bool
                if true, locate will return all matches instead of just the
                best one
            method : int
                one of the cv2.TM_* constants, see cv2.matchTemplate doc for
                info.  you will generally want the normalized methods
                NOTE: TM_CCOEFF_NORMED is *stricter* than the default
                TODO: support TM_SQDIFF*
        """
        super().__init__(**kwargs)
        if target is None:
            raise Exception('Got invalid target for {}'.format(self.name))
        # handle target/grayscale/mode args
        self.grayscale = grayscale
        self.path = None
        if isinstance(target, str):
            self.path = target
            target = cv.imread(target, cv.IMREAD_COLOR)
        target = cv.cvtColor(target, cv.COLOR_BGR2RGB)  # enforce RGB
        if grayscale:
            target = _grayscale(target)
        self.target = target

        self.mask = mask
        self.multi = multi
        self.method = method

    def _get_match_template_func(self, img):
        """ Returns a function that runs cv2.matchTemplate and
            applies self.mask if it exists, ignores it if not.
        """
        # input image preprocessing
        if self.grayscale:
            img = _grayscale(img)
        p1, p2 = self.region
        img = img[p1.y: p2.y, p1.x: p2.x]  # crop to this patterns region

        def match_nomask():
            return cv.matchTemplate(img, self.target, self.method)

        def match_mask():
            return cv.matchTemplate(img, self.target, self.method,
                                    self.mask)
        if self.mask:
            return match_mask
        return match_nomask

    def locate(self, img=None):
        """ Find the image pattern.

            Parameters
            ----------
            img : np.ndarray | None
                optional image to find in (aka haystack),
                otherwise capture is used

            Returns
            -------
            matches : [(Point, Point)]
                list of regions where matches occur

        """
        h, w = self.target.shape[:2]
        if img is None:
            img = capture()
        res = self._get_match_template_func(img)()
        # printable = res[::np.sum(res.shape)]
        # logger.debug('search for {} at {} got the following:\n'
        #              '{}'.format(self.name, self.region, printable))

        # match multi or single (default)
        p1, _ = self.region
        matched = []
        if self.multi:
            loc = np.where(res >= self.confidence)
            # convert raw to point and reapply offset (p1)
            points = [(Point(pt[0], pt[1]) + p1) for pt in zip(*loc[::-1])]
            matched = [(pt, pt + Point(pt.x + w, pt.y + h)) for pt in points]
        else:
            _, max_val, _, mloc = cv.minMaxLoc(res)
            # TODO: what if matchTemplate method is TM_SQ*
            mloc += p1  # reapply offset cus we cropped earlier
            if max_val >= self.confidence:
                matched.append((Point(mloc[0], mloc[1]),
                                Point(mloc[0] + w, mloc[1] + h)))
        if len(matched) > 0:
            logger.debug("found pattern {} at {}, > {}"
                         .format(self.name, matched, self.confidence))

        if self.debug or MAXDEBUG:
            drawn_img = img
            for p1, p2 in matched:
                drawn_img = cv.rectangle(img, tuple(p1), tuple(p2),
                                         (255, 0, 0), 2)  # red
            cv.namedWindow("debug", flags=cv.WINDOW_GUI_NORMAL)
            cv.moveWindow("debug", 0, 0)
            cv.imshow("debug", drawn_img)
            cv.waitKey(6000)
            if self.pause_on_debug:
                breakpoint()

        if len(matched) > 0:
            logger.info('Got {} matches for {}.'
                        .format(len(matched), self.name))
        return matched


class PatternList(Pattern):
    """ Operations for collections of Patterns
    """
    def __init__(self, data=[], match_all=False, **kwargs):
        """ Initialize a PatternList.

            Parameters
            ----------
            data : [Pattern1, ...]
                the patterns to include in this pattern list
            match_all : bool
                when to return True on isvisible  REVIEW
        """
        self.data = data
        self.match_all = match_all
        super().__init__(**kwargs)

    def append(self, p: Pattern):
        """ Add a Pattern to the PatternList.
        """
        self.data.append(p)

    def locate_names(self, img=None):
        """ Locate all Patterns in the PatternList.

            Parameters
            ----------
            img : np.ndarray | None
                optional image to find in, otherwise capture is used

            Returns
            -------
            matches : ([str], [(Point, Point)])
                result of all matching patterns in *names, regions* format
        """
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
        """ Same as locate_names minus the names.

            Parameters
            ----------
            img : np.ndarray | None
                optional image to find in, otherwise capture is used


            Returns
            -------
            matches : [(Point, Point)]
                regions of all matched patterns
        """
        matches = self.locate_names(img=img)[1]
        if len(matches) > 0:
            logger.info('Got {} matches for {}.'
                        .format(len(matches), self.name))
        return matches

    def isvisible(self, img=None):
        """ Check if all or any Patterns in the PatternList
            are visible.  Behaviour defined by match_all property.

            Parameters
            ----------
            img : np.ndarray | None
                optional image to find in, otherwise capture is used

            Returns
            -------
            visible : bool
                whether the pattern is visible or not
        """
        bools = [p.isvisible(img=img) for p in self.data]
        if self.match_all:
            return False not in bools
        else:  # match any
            return True in bools


class ColorPattern(Pattern):
    """ Pattern for finding a cluster of colored points.
    """
    def __init__(self, cts, cluster=5, min_thresh=50, max_thresh=5000,
                 **kwargs):
        """ Initialize a ColorPattern.

            Parameters
            ----------
            cts : CTS
                the *color tolerance speed (aka method)*
            cluster : int = 5 (pixels)
                the radius to use when clustering
            min_thresh : int
                the minimum points for a cluster to be valid
            max_thresh : int
                the maximum points for a cluster to be valid
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

            Parameters
            ----------
            img : np.ndarray | None
                optional image to find in, otherwise capture is used

            Returns
            -------
            points : [[Point, ...]]
                the found points clustered into a 2d array
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
            logger.debug("failed to find {}".format(self.name))
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
        """ Same as locate_points except returns regions instead.

            Parameters
            ----------
            img : np.ndarray | None
                optional image to find in, otherwise capture is used

            Returns
            -------
            regions : [(Point, Point), ...]
                regions of all clusters found by locate_points
        """
        points = self.locate_points(img=img)
        # REVIEW: does this work ?  points sorted or no ?
        regions = [(Point(cl[0][0], cl[0][1]),
                    Point(cl[-1][0], cl[-1][1]))
                   for cl in points]
        if len(regions) > 0:
            logger.info('Got {} matches for {}.'
                        .format(len(regions), self.name))
        return regions
