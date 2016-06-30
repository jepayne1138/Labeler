import configparser
import re
import random
from collections import OrderedDict


class Config:

    """Config singleton

    Seemed the best way of doing this without using globals
    """

    config_path = 'config.ini'
    LABELS_SECTION = 'Labels'
    HEADER_SECTION = 'Headers'
    OPACITY_SECTION = 'Opacity'
    DEFAULT_OPACITY = 255

    labels = OrderedDict()
    headers = OrderedDict()
    opacity = None
    percent_opacity = None

    @classmethod
    def read_config(cls, path=None):
        if path is not None:
            cls.config_path = path
        parser = configparser.ConfigParser()
        parser.read(cls.config_path)

        # Get opacity
        try:
            cls.opacity = int(
                parser[cls.OPACITY_SECTION].get('alpha', cls.DEFAULT_OPACITY)
            )
        except KeyError:
            # Keep as default value
            cls.opacity = cls.DEFAULT_OPACITY
        cls.percent_opacity = cls.opacity / 255

        # Get label
        for k, v in parser[cls.LABELS_SECTION].items():
            rgb = TextColorMap.hex_to_rgb(v)
            if rgb is not None:
                cls.labels[k] = rgb + (cls.percent_opacity,)
        # cls.labels = dict(parser[cls.LABELS_SECTION])

        # Get headers
        for k, v in parser[cls.HEADER_SECTION].items():
            # Keep as a dict for now in case we want to use values later
            cls.headers[k] = v

    @classmethod
    def initialize_config(cls, path=None):
        cls.read_config(path)


class TextColorMap:

    """Maps best text color for given background colors

    Private methods assume validated input, public methods can be made
    that would expose validation methods

    Formulas from: http://stackoverflow.com/a/3118280
    User: https://stackoverflow.com/users/33086/michael-zuschlag
    """

    _map = {}

    # sRGB constants
    S_RED = 0.2126
    S_BLUE = 0.0722
    S_GREEN = 0.7152

    # Default colors
    BLACK = '000000'
    WHITE = 'ffffff'

    @classmethod
    def get(cls, key):
        return cls._map[key]
        # parsed_key = cls.hex_color(key)
        # if parsed_key is None:
        #     return None
        # return cls._map[parsed_key]

    @classmethod
    def _get(cls, key):
        return cls._map[key]

    @classmethod
    def set(cls, key):
        # parsed_color = cls.hex_color(key)
        # if parsed_color is None:
        #     return False
        # cls._map[parsed_color] = cls._best_text_color(parsed_color)
        cls._map[key] = cls._best_text_color(key)

    @staticmethod
    def hex_color(color):
        match = re.search('(\\b#?[A-F0-9]{6}\\b)', color, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    @classmethod
    def best_text_color(cls, color):
        parsed_color = cls.hex_color(color)
        if parsed_color is None:
            return None
        return cls._best_text_color(parsed_color)

    @classmethod
    def _best_text_color(cls, color):
        """Assumes color already validated by hex_color method

        Returns:
            str - Either 'ffffff' or '000000' based on if black or white
                text would provide more contrast to the given color
        """
        return '#{}'.format(cls.BLACK)
        # black = cls._contrast(color, cls.BLACK)
        # white = cls._contrast(color, cls.WHITE)
        # return '#{}'.format(cls.BLACK if black >= white else cls.WHITE)

    @classmethod
    def hex_to_rgb(cls, color):
        parsed_color = cls.hex_color(color)
        if parsed_color is None:
            return None
        return cls._hex_to_rgb(parsed_color)

    @staticmethod
    def _hex_to_rgb(color):
        return (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16)
        )

    @classmethod
    def _luminance(cls, red, green, blue):
        r = (red / 255) ** (2.2)
        g = (green / 255) ** (2.2)
        b = (blue / 255) ** (2.2)
        return (cls.S_RED * r) + (cls.S_GREEN * g) + (cls.S_BLUE * b)

    @classmethod
    def _contrast(cls, color1, color2):
        y1 = cls._luminance(*cls._hex_to_rgb(color1)) + 0.05
        y2 = cls._luminance(*cls._hex_to_rgb(color2)) + 0.05
        # Return the larger divided by smaller (brighter by darker)
        return y1 / y2 if y1 > y2 else y2 / y1


def random_color(offset=80):
    """Maybe better placed elsewhere, but color is config, so for now...

    offset offsets the random color from white or black by a given
    amount, given in the sum of the decimal rgb values
    """
    r = random.randrange(0, 256)
    g = random.randrange(0, 256)
    b_min = max((offset - r - g), 0)
    b_max = min(((256 - offset) + r + g), 256)
    b = random.randrange(b_min, b_max)
    return hex((r << 16) + (g << 8) + b)[2:]
