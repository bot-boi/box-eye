from .botutils import android


# TODO: require certain parameters to be present in injected deps
#       for capture, click, and drag ?  how to do that ?
def capture(*args, **kwargs):
    """ Get a frame from the target device/application.
    """
    return android.capture(*args, **kwargs)


def click(*args, **kwargs):
    """ Click a point on the screen.
        Accepts `region` as well.
    """
    return android.click(*args, **kwargs)


def drag(*args, **kwargs):
    """ Drag cursor from A to B for duration `ms`.
    """
    return android.drag(*args, **kwargs)


def keypress(*args, **kwargs):
    """ Send a key event.
    """
    return android.keypress(*args, **kwargs)
