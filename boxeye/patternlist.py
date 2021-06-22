import logging

from .controls import capture


logger = logging.getLogger('boxeye')


class PatternList():
    """ Operations for collections of Patterns.
        A PatternList implements the standard pattern interface
        but it is not a child of Pattern.
    """
    def __init__(self, *patterns, match_all=False):
        """ Initialize a PatternList.

            Parameters
            ----------
            *patterns : Pattern1, ...
                the patterns to include in this pattern list
            match_all : bool
                return true when all match or any match (isvisible)
        """
        self.data = list(patterns)
        self.match_all = match_all

    def append(self, p):
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
            img = capture()  # avoid getting a new frame for each pattern
        names = []
        loc = []
        for pattern in self.data:
            p_loc = pattern.locate(img=img)
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
