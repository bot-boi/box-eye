# TODO: functional rewrite
# TODO: cleanup code related to the `debug` property
import logging
# TODO: decouple boxeye and botutils
#       is this really necessary?  boxeye has no plans for
#       anything except targeting android via ADB.
from .botutils import android
from .controls import capture, click, drag, keypress
from .cts import CTS2
from .pattern import Pattern
from .patternlist import PatternList
from .point import Point
from .imagepattern import ImagePattern
from .textpattern import TextPattern


logging.getLogger('boxeye').setLevel(logging.INFO)
