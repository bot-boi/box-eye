import cv2 as cv
import logging
import numpy as np

from .botutils.android import capture

from .constants import DEBUG_WAIT_TIME
from .pattern import Pattern
from .point import Point
from .util import grayscale as _grayscale


logger = logging.getLogger('boxeye')


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

        if self.debug:
            drawn_img = img
            for p1, p2 in matched:
                drawn_img = cv.rectangle(img, tuple(p1), tuple(p2),
                                         (255, 0, 0), 2)  # red
            cv.namedWindow("debug", flags=cv.WINDOW_GUI_NORMAL)
            cv.moveWindow("debug", 0, 0)
            cv.imshow("debug", drawn_img)
            cv.waitKey(DEBUG_WAIT_TIME)
            if self.pause_on_debug:
                breakpoint()

        if len(matched) > 0:
            logger.info('Got {} matches for {}.'
                        .format(len(matched), self.name))
        return matched
