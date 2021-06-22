import logging
import numpy as np

from .cts import CTS2
from .pattern import Pattern
from .point import Point


logger = logging.getLogger('boxeye')


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
