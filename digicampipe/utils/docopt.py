import numpy as np


def convert_list_int(text):
    list_int = None
    if text is not None:
        text = text.split(',')
        list_int = list(map(int, text))
        list_int = np.array(list_int)
    return list_int


def convert_list_float(text):
    list_float = None
    if text is not None:
        text = text.split(',')
        list_float = list(map(float, text))
        list_float = np.array(list_float)
    return list_float


def convert_int(text):
    max_events = None
    if text is not None:
        max_events = int(text)
    return max_events


def convert_pixel_args(text):
    pixel_id = convert_list_int(text)
    if pixel_id is None:
        pixel_id = np.arange(1296)
    return pixel_id



