import numpy as np


def convert_text(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        return text


def convert_int(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        return int(text)


def convert_float(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        return float(text)


def convert_list_int(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        text = text.split(',')
        list_int = list(map(int, text))
        return np.array(list_int)


def convert_list_float(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        text = text.split(',')
        list_float = list(map(float, text))
        return np.array(list_float)


def convert_pixel_args(text):
    pixel_id = convert_list_int(text)
    if pixel_id is None:
        pixel_id = np.arange(1296)
    return pixel_id
