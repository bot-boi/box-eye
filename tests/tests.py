import os
from src import util, TextPattern


check_vision = util.make_check_vision(os.path.dirname(__file__) + os.sep)


# TODO: better testing
def test_txt_pattern():
    p = TextPattern("RECONNECT")
    check_vision(p, "connection-error", "game-idle")


# def test_img_pattern():
#     check_vision(game.chat, "game-idle", "menu")


# def test_img_pattern_list():
#     check_vision(game.workout, "workout", "menu")
