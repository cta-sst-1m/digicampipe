import numpy as np


def convert_pixel_args(text):

    if text is not None:

        text = text.split(',')
        pixel_id = list(map(int, text))
        pixel_id = np.array(pixel_id)

    else:

        pixel_id = np.arange(1296)

    return pixel_id


def convert_dac_level(text):

    if text is not None:

        text = text.split(',')
        dac_level = list(map(int, text))
        dac_level = np.array(dac_level)

    return dac_level


def convert_max_events_args(text):

    if text is not None:

        max_events = int(text)

    else:

        max_events = text

    return max_events
