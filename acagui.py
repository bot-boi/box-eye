""" acagui.py is a gui tool for generating boxeye Patterns.
"""
import base64
import logging
from time import sleep

import cv2 as cv
import numpy as np
import ppadb.client
import pyperclip
import PySimpleGUI as sg

from boxeye import capture, ColorPattern, ImagePattern, Point, TextPattern
from boxeye.botutils.extra import get_app_path
# from boxeye.botutils import debug_mode
from boxeye.cts import CTS2


logging.root.setLevel(logging.DEBUG)
logging.getLogger('boxeye').setLevel(logging.DEBUG)


MODE_COLORPATTERN = "colpat"
MODE_COLORPATTERN_EVENT = "ColorPattern::{}".format(MODE_COLORPATTERN)
MODE_TEXTPATTERN = "txtpat"
MODE_TEXTPATTERN_EVENT = "TextPattern::{}".format(MODE_TEXTPATTERN)
MODE_IMAGEPATTERN = "imgpat"
MODE_IMAGEPATTERN_EVENT = "ImagePattern::{}".format(MODE_IMAGEPATTERN)


client = ppadb.client.Client()
logging.root.setLevel("DEBUG")


DATA_DIR = get_app_path(__file__)


def encode_base64(fpath):
    """ open an image file and encode as base64

        Parameters:
        -----------
            fpath - the path of the target file

        Returns:
        ________
            a base64 string representation of file

    """
    with open(fpath, "rb") as f:
        res = base64.b64encode(f.read())
    return res


def decode_base64(img_str):
    """ convert a base64 str to opencv image

        Parameters:
        -----------
            img_str - the encoded image

        Returns:
        ________
            an opencv image (numpy array)

    """
    decode_b64 = base64.b64decode(img_str)
    raw = np.fromstring(decode_b64, dtype='uint8')
    return cv.imdecode(raw, cv.IMREAD_COLOR)


def image_to_string(img):
    """ Converts an image for display in ACA's Graph element (the image viewer)

        Parameters:
        ___________
            img: the image (numpy array) to display

        Returns:
        ________
            byte string representation of input image
    """
    return cv.imencode('.png', img)[1].tobytes()


class AutoColorAid:
    """ ACA GUI !!! """
    def __init__(self, default_img_path=None):

        # buttons and stuff

        # always visible
        actions_menu = [
            [sg.Button("capture"), sg.Button("exit")],
            [sg.Button("draw"), sg.Button("erase")],
            [sg.FileBrowse("load"),
             sg.In(key="file-load", enable_events=True)],
            [sg.Button("clipboard")],
            [sg.ButtonMenu("mode",
                           ["",
                            [MODE_COLORPATTERN_EVENT,
                             MODE_IMAGEPATTERN_EVENT,
                             MODE_TEXTPATTERN_EVENT]],
                           key="mode"),
             sg.Text("", size=(10, 1), key="mode-out")],
        ]
        pattern_menu = [
            # [sg.In(default_text="192.168.1.138:9999", key="-TARGET-DEVICE-")]
            [sg.Multiline(size=(None, 3), key="pattern-out")],
            [sg.In(tooltip="confidence", default_text="0.6",
                   key="confidence")],
            [sg.In(tooltip="region", default_text="0 0, -1 -1",
                   key="region")],
            [sg.In(default_text="NEW-ACA-PATTERN", key="pattern-name")],
            [sg.Check('debug', key='pattern-debug')],
            [sg.Check('pause on debug', key='pattern-pause-on-debug')],
        ]
        primary_menu = sg.Frame("primary menu", actions_menu + pattern_menu)
        colpat_menu = \
            sg.pin(sg.Column([
                        [sg.In(tooltip="clustering radius",
                               default_text="5",
                               key="colpat-cluster-radius")],
                        [sg.In(tooltip="min pts per blob",
                               default_text="10",
                               key="colpat-min-thresh")],
                        [sg.In(tooltip="max pts per blob",
                               default_text="10000",
                               key="colpat-max-thresh")],
                        [sg.Check("pick color", key="pick-colors")],
                    ],
                    key=MODE_COLORPATTERN, visible=False))
        imgpat_menu = \
            sg.pin(sg.Column([[sg.Button("set needle", key="imgpat-needle-set",
                                  tooltip="cut img to current region and use" \
                                          " as search needle"
                                  )],
                       [sg.Check("grayscale", key="imgpat-grayscale")],
                       [sg.Graph(background_color="red", key="imgpat-needle-preview",
                                 graph_top_right=(2000, 0),
                                 graph_bottom_left=(0, 2000),
                                 canvas_size=(300, 300))]],
                      key=MODE_IMAGEPATTERN, visible=True))
        txtpat_menu = \
            sg.pin(sg.Column([
                        [sg.In(key="txtpat-pattern-text",
                               tooltip="text to search for (regex)")],
                        [sg.In(default_text="10", key="txtpat-scale",
                               tooltip="scale up by")],
                        [sg.In(default_text="150",
                               key="txtpat-binary-threshold",
                               tooltip="threshold to binarize by")],
                        [sg.Check("invert", key="txtpat-invert",
                                  tooltip="invert the binary image")],
                        [sg.In(default_text="", key="txtpat-config",
                               tooltip="args for tesseract ocr")]
                    ], key=MODE_TEXTPATTERN, visible=False))

        # color list
        colors_mode = sg.LISTBOX_SELECT_MODE_MULTIPLE
        color_menu = [[], ['Delete::color-list']]
        color_list = sg.Listbox(key="color-list", select_mode=colors_mode,
                                size=(18, 18), values=[],
                                right_click_menu=color_menu)

        # main screen
        img_elem = sg.Graph(key="imgview", enable_events=True,
                            graph_top_right=(2000, 0),
                            graph_bottom_left=(0, 2000),
                            canvas_size=(2000, 2000))
        img_viewer = sg.Column([[img_elem]], size=(1000, 700), scrollable=True)

        interface = sg.Column([
            [primary_menu],
            [imgpat_menu],
            [txtpat_menu],
            [colpat_menu],
            [color_list]
        ])
        layout = [[img_viewer, interface]]
        window = sg.Window("ACA.py: Auto Color Aid for boxeye", layout,
                           location=(0, 20), size=(1366, 748))
        self.colors = []  # MODE_COLORPATTERN
        self.current_img = None
        self.current_img_path = None
        self.mode = MODE_IMAGEPATTERN
        self.needle_path = DATA_DIR + "needle.png"  # MODE_IMAGEPATTERN
        self.needle_region = []  # MODE_IMAGEPATTERN
        window.read(timeout=1)
        window['mode-out'].update(value=MODE_IMAGEPATTERN)
        self.region = []
        self.window = window

        self.window.read(timeout=1)
        if default_img_path:
            self.current_img = cv.imread(default_img_path)
            self.current_img_path = default_img_path
            self.current_img_clear()

    def _event_capture(self, values):
        """ capture the screen
        """
        raise NotImplementedError

    def _event_clipboard(self, values):
        """ copy to clipboard, *clipboard* btn
        """
        if len(self.colors) > 0 and self.window.mode == MODE_COLORPATTERN:
            r = CTS2.from_colors(self.colors)
            resultstr = 'CTS2({}, {}, {}, {}, {}, {})' \
                        .format(r.r, r.g, r.b, r.rtol, r.gtol, r.btol)
            pyperclip.copy(resultstr)

    def _draw_regions(self, regions):
        """ draw regions returned by boxeye Pattern.locate
            onto the image viewer
        """
        for p1, p2 in regions:
            self.window['imgview'] \
                .draw_rectangle(top_left=p1, bottom_right=p2,
                                line_color="red")

    def _event_draw_colpat(self, values):
        """ MODE_COLORPATTERN draw event
            this is where all the finding/generating happens
        """
        # TODO: display CTS color range in preview
        cluster = int(self.window['colpat-cluster-radius'].get())
        mf = int(self.window['colpat-min-thresh'].get())
        Mf = int(self.window['colpat-max-thresh'].get())
        cts = CTS2.from_colors(self.colors)
        pattern = ColorPattern(cts, cluster=cluster,
                               min_thresh=mf, max_thresh=Mf,
                               **self.get_pattern_menu_stuff())
        conf = self.get_pattern_menu_stuff()['confidence']
        name = self.get_pattern_menu_stuff()['name']
        pattern_str = "CPat({}, confidence={},\n" \
                      "     region={}, cluster={},\n" \
                      "     min_thresh={}, max_thresh={},\n" \
                      "     name='{}')\n" \
                      .format(cts, conf,
                              self._stringify_region(self.region),
                              cluster, mf, Mf, name)
        self.window["pattern-out"].update(pattern_str)
        result = pattern.locate(img=self.current_img)
        self._draw_regions(result)

    def _event_draw_imgpat(self, values):
        """ MODE_IMAGEPATTERN draw event
            this is where all the finding/generating happens
        """
        # TODO: file save dialog for needle
        grayscale = bool(self.window['imgpat-grayscale'].get())
        pattern = ImagePattern(self.needle_path, grayscale=grayscale,
                               **self.get_pattern_menu_stuff())

        conf = self.get_pattern_menu_stuff()['confidence']
        name = self.get_pattern_menu_stuff()['name']
        pattern_str = "IPat({}, region={},\n" \
                      "     confidence={}, grayscale={},\n" \
                      "     name='{}')\n" \
                      .format(self.needle_path,
                              self._stringify_region(self.region),
                              conf, grayscale, name)
        self.window["pattern-out"].update(pattern_str)
        result = pattern.locate(img=self.current_img)
        self._draw_regions(result)

    def get_pattern_menu_stuff(self):
        return {
            'region': self.region,
            'confidence': float(self.window['confidence'].get()),
            'debug': self.window['pattern-debug'].get(),
            'pause_on_debug': self.window['pattern-pause-on-debug'].get(),
            'name': self.window['pattern-name'].get(),
        }

    def _event_draw_txtpat(self, values):
        """ MODE_TEXTPATTERN draw event
            this is where all the finding/generating happens
        """
        # TODO: preview graph, show img post processing
        pattern_text = self.window['txtpat-pattern-text'].get()
        scale = int(self.window['txtpat-scale'].get())
        invert = bool(self.window['txtpat-invert'].get())
        thresh = int(self.window['txtpat-binary-threshold'].get())
        config = self.window['txtpat-config'].get()
        pattern = TextPattern(pattern_text, scale=scale, invert=invert,
                              threshold=thresh, config=config,
                              **self.get_pattern_menu_stuff())

        conf = self.get_pattern_menu_stuff()['confidence']
        name = self.get_pattern_menu_stuff()['name']
        pattern_str = "TPat('{}', region={},\n" \
                      "     confidence={}, invert={},\n" \
                      "     scale={}, threshold={},\n" \
                      "     config={}, name='{}')\n" \
                      .format(pattern_text, self._stringify_region(self.region),
                              conf, invert, scale, thresh, config, name)
        self.window["pattern-out"].update(pattern_str)
        result = pattern.locate(img=self.current_img)
        logging.debug(result)
        self._draw_regions(result)

    def _event_draw(self, values):
        """ handle draw event (button *draw*)
            run the appropriate draw event for the current mode
        """
        if self.mode == MODE_COLORPATTERN:
            self._event_draw_colpat(values)
        elif self.mode == MODE_IMAGEPATTERN:
            self._event_draw_imgpat(values)
        elif self.mode == MODE_TEXTPATTERN:
            self._event_draw_txtpat(values)

    def _event_erase(self, values):
        """ remove marks from the image view """
        self.current_img_clear()

    def _event_file_load(self, values):
        """ user load file (button *load*)
            only works with .png
        """
        fpath = self.window["file-load"].get()
        logging.debug("Loading file {}".format(fpath))

        img = cv.imread(fpath)
        self.current_img = img
        self.current_img_clear()

    def _event_click_graph(self, values):
        """ click event for image viewer """
        pos = values['imgview']
        x, y = pos
        if self.mode == MODE_COLORPATTERN and \
                self.window['pick-colors'].get() == 1:
            # pick colors for color tolerance generation (CTS2)
            color = CTS2(self.current_img[y][x], 0, 0, 0)
            self.colors.append(color)
            self.window['color-list'].update(values=[c.asarray()[:3]
                                                     for c in self.colors])
        else:
            # select region
            pt = Point(x, y)
            rlen = len(self.region)
            if rlen >= 2 or rlen <= 0:  # reset
                self.region = []
                self.region.append(pt)
            elif rlen == 1:  # add second point
                self.region.append(pt)
                region_str = self._stringify_region(self.region)
                self.window['region'].update(value=region_str)

    def _event_mode_change(self, values):
        """ triggered when user changes mode (multibutton *mode*) """
        mode = values['mode']
        self.mode = mode
        if mode == MODE_COLORPATTERN_EVENT:
            self.mode = MODE_COLORPATTERN
            self.window[MODE_IMAGEPATTERN].update(visible=False)
            self.window[MODE_TEXTPATTERN].update(visible=False)
            self.window[MODE_COLORPATTERN].update(visible=True)
            self.window['mode-out'].update(value=MODE_COLORPATTERN)
        elif mode == MODE_IMAGEPATTERN_EVENT:
            self.mode = MODE_IMAGEPATTERN
            self.window[MODE_COLORPATTERN].update(visible=False)
            self.window[MODE_TEXTPATTERN].update(visible=False)
            self.window[MODE_IMAGEPATTERN].update(visible=True)
            self.window['mode-out'].update(value=MODE_IMAGEPATTERN)
        elif mode == MODE_TEXTPATTERN_EVENT:
            self.mode = MODE_TEXTPATTERN
            self.window[MODE_COLORPATTERN].update(visible=False)
            self.window[MODE_IMAGEPATTERN].update(visible=False)
            self.window[MODE_TEXTPATTERN].update(visible=True)
            self.window['mode-out'].update(value=MODE_TEXTPATTERN)

    def _event_imgpat_needle_set(self, values):
        """ set the image search needle to the
            currently selected region (of haystack).

            MODE_IMAGEPATTERN, imgpat-needle-set
        """
        if len(self.region) != 2:
            logging.debug("select region first")
            return
        self.needle_region = self.region
        p1, p2 = self.needle_region
        img = self.current_img[p1.y: p2.y, p1.x: p2.x]
        cv.imwrite(self.needle_path, img)
        # col, row, dep = img.shape
        # img = img.reshape((col * row, dep))  # reshape for encoding
        # img_str = base64.b64encode(img.tobytes())
        img_str = image_to_string(img)
        self.window['imgpat-needle-preview'].erase()
        self.window['imgpat-needle-preview'].draw_image(location=(0, 0), data=img_str)

    def _parse_region(self, region_str):
        """ convert user region from '(P(x1, y1), P(x2, y2))' to (Point, Point) """
        region = []
        last_char = None
        delete_me = ['(', ')', 'P', ' ']
        for i in delete_me:
            region_str = region_str.replace(i, '')
        rr = region_str.split(',')  # region raw
        ri = [int(i) for i in rr]  # region integer
        region = (Point(ri[0], ri[1]), Point(ri[2], ri[3]))
        return region

    def _stringify_region(self, region):
        """ Convert region from (Point, Point) to '(P(x1, y1), P(x2, y2))'
        """
        p1, p2 = region
        return "(P({}, {}), P({}, {}))".format(p1.x, p1.y, p2.x, p2.y)

    def current_img_clear(self):
        """ clear the image viewer of all marks and redraw """
        self.window['imgview'].erase()
        img_str = image_to_string(self.current_img)
        self.window['imgview'].draw_image(data=img_str, location=(0, 0))

    def handle_event(self, event, values):
        """ event -> handler mapping """
        event_handlers = {
            '__TIMEOUT__': lambda donothing: None,
            'capture': self._event_capture,
            'clipboard': self._event_clipboard,
            'draw': self._event_draw,
            'erase': self._event_erase,
            'exit': lambda values: self.window.close(),
            None: lambda values: self.window.close(),
            'file-load': self._event_file_load,
            'imgpat-needle-set': self._event_imgpat_needle_set,
            'imgview': self._event_click_graph,
            'mode': self._event_mode_change,
        }
        try:
            event_handlers[event](values)
        except Exception as err:
            raise err
            # logging.debug(err)

    def run(self):
        """ ACA event loop """
        while True:
            event, values = self.window.read(timeout=10000)
            if event is not None:
                logging.debug("EVENT: {}".format(event))
                # logging.debug("VALUES: {}".format(values))
            elif event == sg.WIN_CLOSED:
                break
            self.handle_event(event, values)
            sleep(0.1)
        self.window.close()


if __name__ == "__main__":
    # TODO: include a default image
    aca = AutoColorAid(default_img_path='./OSRS-Banner.jpg')
    aca.run()
