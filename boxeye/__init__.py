# TODO: functional rewrite
# TODO: cleanup code related to the `debug` property
import logging
# TODO: decouple boxeye and botutils
#       is this really necessary?  boxeye has no plans for
#       anything except targeting android via ADB.
from .botutils import android
from .controls import capture, click, drag, keypress
from .cts import CTS2
from .point import Point

from .pattern import Pattern
from .patternlist import PatternList
from .colorpattern import ColorPattern
from .imagepattern import ImagePattern
from .textpattern import TextPattern


logging.getLogger('boxeye').setLevel(logging.INFO)
