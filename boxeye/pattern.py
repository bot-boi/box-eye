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
