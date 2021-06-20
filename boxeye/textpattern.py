import re
import logging

import cv2 as cv
import pytesseract

from .botutils.android import capture

from .constants import DEBUG_WAIT_TIME
from .pattern import Pattern
from .point import Point
from .util import binarize as _binarize


logger = logging.getLogger('boxeye')


# NOTE: also includes NumberReader
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
                                         output_type=pytesseract.Output.DICT)
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
        if self.debug:
            cv.namedWindow("debug", flags=cv.WINDOW_GUI_NORMAL)
            cv.moveWindow("debug", 0, 0)
            cv.imshow("debug", whole_img)
            cv.waitKey(DEBUG_WAIT_TIME)
            cv.imshow("debug", img)
            cv.waitKey(DEBUG_WAIT_TIME)
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
        Meant for reading a single number from a region on the screen,
        useful for checking your characters hit points for example.
    """
    def __init__(self, target, config="--oem 0 --psm 8 -c "
                 "tessedit_char_whitelist=,.0123456789KM", **kwargs):
        super().__init__(target, config=config, **kwargs)

    def get(self, img=None):
        """ Locate and convert found text to number.
        """
        raw = self.locate_names(img=img)  # [((p1, p2), text)]
        if len(raw) <= 0:
            return None
        raw_strings = [i[1] for i in raw]
        text = raw_strings[0]  # TODO: pick match with highest confidence?
        # cleanup
        text = text.strip()
        text = text.replace(',', '')
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
