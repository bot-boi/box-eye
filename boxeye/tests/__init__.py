import os

from .. import ImagePattern as IPat
from .. import Point as P
from .. import TextPattern as TPat
from ..botutils import make_check_vision


path = os.path.dirname(__file__) + os.sep
check_vision = make_check_vision(path)


def test_txt_pattern():
    p = TPat("RECONNECT", region=(P(0, 0), P(959, 539)),
             confidence=0.6, config="")
    check_vision(p, "isvisible", "connection-error")
    check_vision(p, "isvisible", "game-idle", expected=False)


def test_img_pattern():
    p = IPat(path + "chat.png", region=(P(0, 438), P(95, 539)),
             name="CHAT", confidence=0.8)
    check_vision(p, "isvisible", "bet-350k", "game-idle")
    check_vision(p, "isvisible", "connection-error", expected=False)


# def test_img_pattern_list():
#     check_vision(game.workout, "workout", "menu")
