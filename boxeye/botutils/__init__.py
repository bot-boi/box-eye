import argparse
import logging
import sched
import pytest
import cv2

from .extra import periodic


class ColorBot():
    def __init__(self, logger=logging):
        self.id = None
        self.log = logger
        self.parser = argparse.ArgumentParser()
        self.scheduler = sched.scheduler()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def init(self):
        raise NotImplementedError

    # @retry()
    def run(self):
        self.scheduler.run()

    def enter_periodic(self, *args, **kwargs):
        periodic(self.scheduler, *args, **kwargs)


def make_check_vision(imgpath):
    """ Make a *check_vision* function for use with tests """
    def check_vision(obj, method_name, *fnargs, expected=True, methodargs=(),
                     **kwargs):
        """ Check some computer vision action that returns a result.
              obj.method_name should accept a keyword argument *img*,
              else it won't work.

            Parameters
            ----------
                obj - the object to test
                method_name - name of the method of the object to test
                fnargs - list of images to run on
                expected - the value that obj.method_name should return
                methodargs - positional args for obj.method_name
                kwargs - keyword args for obj.method_name
        """

        imgs = list(fnargs)
        # process image names
        imgs = [imgpath + img_name for img_name in imgs]
        imgs = [img_name + ".png" for img_name in imgs]
        imgs = [(cv2.imread(img_name, cv2.IMREAD_COLOR), img_name)
                for img_name in imgs]
        method = getattr(obj, method_name)
        for img, img_name in imgs:
            if not (method(img=img, *methodargs, **kwargs) == expected):
                pytest.fail("{} failed on {}".format(method_name, img_name))
        obj.debug = False

    return check_vision
