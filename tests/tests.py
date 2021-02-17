import os
from boxeye import TextPattern
from PIL import Image
import pytest
from vectormath import Vector2 as P


# OLD, see android-automation for new check_vision
def make_check_vision(path):
    def check_vision(pattern, expect_true, expect_false=[]):
        if isinstance(expect_true, str):
            expect_true = [expect_true]
        if isinstance(expect_false, str):
            expect_false = [expect_false]

        # add .png extension if not present and convert to Image
        expect_true = list(map(lambda i: i + ".png" if ".png" not in i else i,
                               expect_true))
        expect_false = list(map(lambda i: i + ".png" if ".png" not in i else i,
                                expect_false))

        # UGLY
        # do the tests
        for fname in expect_true:
            img = Image.open(path + fname)
            res = pattern.isvisible(img=img)
            if res is not True:
                pytest.fail("uh oh {} is not visible in {}"
                            .format(pattern.name, fname))
            assert res is True
        for fname in expect_false:
            img = Image.open(path + fname)
            res = pattern.isvisible(img=img)
            if res is not False:
                pytest.fail("uh oh {} is visible in {}"
                            .format(pattern.name, fname))
            assert res is False

    return check_vision


check_vision = make_check_vision(os.path.dirname(__file__) + os.sep)


# TODO: better testing
def test_txt_pattern():
    # init(None)
    p = TextPattern("RECONNECT", region=(P(0, 0), P(959, 539)), config="")
    check_vision(p, "connection-error", "game-idle")


# def test_img_pattern():
#     check_vision(game.chat, "game-idle", "menu")


# def test_img_pattern_list():
#     check_vision(game.workout, "workout", "menu")
