import game
import util
from constants import APP_PATH

import core

path = APP_PATH + "screencaps/"


# ================ TESTS ==================
# very basic
# TODO: dont import from game, use more generic tests
#   for when we move this out of this project
check_vision = util.make_check_vision(APP_PATH + "screencaps/")


def test_txt_pattern():
    p = core.TextPattern("RECONNECT")
    check_vision(p, "connection-error", "game-idle")


def test_img_pattern():
    check_vision(game.chat, "game-idle", "menu")


def test_img_pattern_list():
    check_vision(game.workout, "workout", "menu")
