import argparse
import logging
import sched
from retry import retry

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

    @retry()
    def run(self):
        self.scheduler.run()

    def enter_periodic(self, *args, **kwargs):
        periodic(self.scheduler, *args, **kwargs)


def get_android_cbot():
    from .android import capture, click, drag
    cbot = ColorBot()
    setattr(cbot, "capture", capture)
    setattr(cbot, "click", click)
    setattr(cbot, "drag", drag)
    return cbot
