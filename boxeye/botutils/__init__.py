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
