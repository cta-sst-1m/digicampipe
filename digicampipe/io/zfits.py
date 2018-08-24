# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.
This requires the protozfits python library to be installed
"""
import logging
from tqdm import tqdm
import numpy as np
import warnings
from digicampipe.io.containers import DataContainer
import digicampipe.utils as utils
from protozfits import File
logger = logging.getLogger(__name__)


__all__ = ['zfits_event_source']


def zfits_event_source(
    url,
    camera=utils.DigiCam,
    max_events=None,
    allowed_tels=None,
):
    """A generator that streams data from an ZFITs data file
    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : list[int]
        select only a subset of telescope, if None, all are read. This can
        be used for example emulate the final CTA data format, where there
        would be 1 telescope per file (whereas in current monte-carlo,
        they are all interleaved into one file)
    camera : utils.Camera(), default DigiCam
    """
    data = DataContainer()
    with File(url) as file:

        for event_counter, event in tqdm(
            enumerate(file.Events),
            desc='Events',
            leave=True,
            total=len(file.Events),
        ):
            if max_events is not None and event_counter > max_events:
                break

            data.r0.event_id = event_counter
            data.r0.tels_with_data = [event.telescopeID, ]

            # remove forbidden telescopes
            if allowed_tels:
                data.r0.tels_with_data = [
                    list(
                        filter(
                            lambda x: x in data.r0.tels_with_data, sublist))
                    for sublist in allowed_tels
                ]

            _sort_ids = None

            for tel_id in data.r0.tels_with_data:

                pixel_ids = event.hiGain.waveforms.pixelsIndices
                n_pixels = len(pixel_ids)
                if _sort_ids is None:
                    _sort_ids = np.argsort(pixel_ids)
                samples = event.hiGain.waveforms.samples.reshape(n_pixels, -1)

                try:
                    unsorted_baseline = event.hiGain.waveforms.baselines
                except AttributeError:
                    warnings.warn((
                        "Could not read `hiGain.waveforms.baselines`"
                        " for event:{0}"
                        "of file:{url}".format(event_counter, url)
                        ))
                    return np.ones(n_pixels) * np.nan

                data.inst.num_channels[tel_id] = event.num_gains
                data.inst.geom[tel_id] = camera.geometry
                data.inst.cluster_matrix_7[tel_id] = camera.cluster_7_matrix
                data.inst.cluster_matrix_19[tel_id] = camera.cluster_19_matrix
                data.inst.patch_matrix[tel_id] = camera.patch_matrix
                data.inst.num_pixels[tel_id] = samples.shape[0]
                data.inst.num_samples[tel_id] = samples.shape[1]

                r0 = data.r0.tel[tel_id]
                r0.camera_event_number = event.eventNumber
                r0.pixel_flags = event.pixels_flags[_sort_ids]
                r0.local_camera_clock = (
                    event.local_time_sec * 1E9 + event.local_time_nanosec
                )
                r0.gps_time = (
                    event.trig.timeSec * 1E9 + event.trig.timeNanoSec
                )
                r0.camera_event_type = event.event_type
                r0.array_event_type = event.eventType
                r0.adc_samples = samples[_sort_ids]

                if len(event.trigger_input_traces) > 0:
                    r0.trigger_input_traces = _prepare_trigger_input(
                        event.trigger_input_traces
                    )
                else:
                    warnings.warn(
                        'trigger_input_traces does not exist: --> nan')
                    r0.trigger_input_traces = np.zeros(
                        (432, data.inst.num_samples[tel_id])) * np.nan

                if len(event.trigger_output_patch7) > 0:
                    r0.trigger_output_patch7 = _prepare_trigger_output(
                        event.trigger_output_patch7)
                else:
                    warnings.warn(
                        'trigger_output_patch7 does not exist: --> nan')
                    r0.trigger_output_patch7 = np.zeros(
                        (432, data.inst.num_samples[tel_id])) * np.nan

                if len(event.trigger_output_patch19) > 0:
                    r0.trigger_output_patch19 = _prepare_trigger_output(
                        event.trigger_output_patch19)
                else:
                    warnings.warn(
                        'trigger_output_patch19 does not exist: --> nan')
                    r0.trigger_output_patch19 = np.zeros(
                        (432, data.inst.num_samples[tel_id])) * np.nan

                r0.digicam_baseline = unsorted_baseline[_sort_ids] / 16

            yield data


def count_number_events(file_list):
    return sum(
        len(File(filename).Events)
        for filename in file_list
    )


PATCH_ID_INPUT = [
    204, 216, 180, 192, 229, 241, 205, 217, 254, 266, 230, 242,
    279, 291, 255, 267, 304, 316, 280, 292, 329, 341, 305, 317, 156, 168, 132,
    144, 181, 193, 157, 169, 206, 218, 182, 194, 231, 243, 207, 219, 256, 268,
    232, 244, 281, 293, 257, 269, 108, 120, 84, 96, 133, 145, 109, 121, 158,
    170, 134, 146, 183, 195, 159, 171, 208, 220, 184, 196, 233, 245, 209, 221,
    60, 72, 40, 50, 85, 97, 61, 73, 110, 122, 86, 98, 135, 147, 111, 123, 160,
    172, 136, 148, 185, 197, 161, 173, 24, 32, 12, 18, 41, 51, 25, 33, 62, 74,
    42, 52, 87, 99, 63, 75, 112, 124, 88, 100, 137, 149, 113, 125, 4, 8, 0, 2,
    13, 19, 5, 9, 26, 34, 14, 20, 43, 53, 27, 35, 64, 76, 44, 54, 89, 101, 65,
    77, 228, 239, 240, 252, 251, 262, 263, 275, 274, 285, 286, 298, 297, 308,
    309, 321, 320, 331, 332, 344, 343, 354, 355, 366, 253, 264, 265, 277, 276,
    287, 288, 300, 299, 310, 311, 323, 322, 333, 334, 346, 345, 356, 357, 368,
    367, 377, 378, 387, 278, 289, 290, 302, 301, 312, 313, 325, 324, 335, 336,
    348, 347, 358, 359, 370, 369, 379, 380, 389, 388, 396, 397, 404, 303, 314,
    315, 327, 326, 337, 338, 350, 349, 360, 361, 372, 371, 381, 382, 391, 390,
    398, 399, 406, 405, 411, 412, 417, 328, 339, 340, 352, 351, 362, 363, 374,
    373, 383, 384, 393, 392, 400, 401, 408, 407, 413, 414, 419, 418, 422, 423,
    426, 353, 364, 365, 376, 375, 385, 386, 395, 394, 402, 403, 410, 409, 415,
    416, 421, 420, 424, 425, 428, 427, 429, 430, 431, 215, 191, 227, 203, 167,
    143, 179, 155, 119, 95, 131, 107, 71, 49, 83, 59, 31, 17, 39, 23, 7, 1,
    11, 3, 238, 214, 250, 226, 190, 166, 202, 178, 142, 118, 154, 130, 94, 70,
    106, 82, 48, 30, 58, 38, 16, 6, 22, 10, 261, 237, 273, 249, 213, 189, 225,
    201, 165, 141, 177, 153, 117, 93, 129, 105, 69, 47, 81, 57, 29, 15, 37,
    21, 284, 260, 296, 272, 236, 212, 248, 224, 188, 164, 200, 176, 140, 116,
    152, 128, 92, 68, 104, 80, 46, 28, 56, 36, 307, 283, 319, 295, 259, 235,
    271, 247, 211, 187, 223, 199, 163, 139, 175, 151, 115, 91, 127, 103, 67,
    45, 79, 55, 330, 306, 342, 318, 282, 258, 294, 270, 234, 210, 246, 222,
    186, 162, 198, 174, 138, 114, 150, 126, 90, 66, 102, 78
]

PATCH_ID_INPUT_SORT_IDS = np.argsort(PATCH_ID_INPUT)

PATCH_ID_OUTPUT = [
    204, 216, 229, 241, 254, 266, 279, 291, 304, 316, 329,
    341, 180, 192, 205, 217, 230, 242, 255, 267, 280, 292, 305, 317, 156, 168,
    181, 193, 206, 218, 231, 243, 256, 268, 281, 293, 132, 144, 157, 169, 182,
    194, 207, 219, 232, 244, 257, 269, 108, 120, 133, 145, 158, 170, 183, 195,
    208, 220, 233, 245, 84, 96, 109, 121, 134, 146, 159, 171, 184, 196, 209,
    221, 60, 72, 85, 97, 110, 122, 135, 147, 160, 172, 185, 197, 40, 50, 61,
    73, 86, 98, 111, 123, 136, 148, 161, 173, 24, 32, 41, 51, 62, 74, 87, 99,
    112, 124, 137, 149, 12, 18, 25, 33, 42, 52, 63, 75, 88, 100, 113, 125, 4,
    8, 13, 19, 26, 34, 43, 53, 64, 76, 89, 101, 0, 2, 5, 9, 14, 20, 27, 35,
    44, 54, 65, 77, 228, 239, 251, 262, 274, 285, 297, 308, 320, 331, 343,
    354, 240, 252, 263, 275, 286, 298, 309, 321, 332, 344, 355, 366, 253, 264,
    276, 287, 299, 310, 322, 333, 345, 356, 367, 377, 265, 277, 288, 300, 311,
    323, 334, 346, 357, 368, 378, 387, 278, 289, 301, 312, 324, 335, 347, 358,
    369, 379, 388, 396, 290, 302, 313, 325, 336, 348, 359, 370, 380, 389, 397,
    404, 303, 314, 326, 337, 349, 360, 371, 381, 390, 398, 405, 411, 315, 327,
    338, 350, 361, 372, 382, 391, 399, 406, 412, 417, 328, 339, 351, 362, 373,
    383, 392, 400, 407, 413, 418, 422, 340, 352, 363, 374, 384, 393, 401, 408,
    414, 419, 423, 426, 353, 364, 375, 385, 394, 402, 409, 415, 420, 424, 427,
    429, 365, 376, 386, 395, 403, 410, 416, 421, 425, 428, 430, 431, 215, 191,
    167, 143, 119, 95, 71, 49, 31, 17, 7, 1, 227, 203, 179, 155, 131, 107, 83,
    59, 39, 23, 11, 3, 238, 214, 190, 166, 142, 118, 94, 70, 48, 30, 16, 6,
    250, 226, 202, 178, 154, 130, 106, 82, 58, 38, 22, 10, 261, 237, 213, 189,
    165, 141, 117, 93, 69, 47, 29, 15, 273, 249, 225, 201, 177, 153, 129, 105,
    81, 57, 37, 21, 284, 260, 236, 212, 188, 164, 140, 116, 92, 68, 46, 28,
    296, 272, 248, 224, 200, 176, 152, 128, 104, 80, 56, 36, 307, 283, 259,
    235, 211, 187, 163, 139, 115, 91, 67, 45, 319, 295, 271, 247, 223, 199,
    175, 151, 127, 103, 79, 55, 330, 306, 282, 258, 234, 210, 186, 162, 138,
    114, 90, 66, 342, 318, 294, 270, 246, 222, 198, 174, 150, 126, 102, 78
]

PATCH_ID_OUTPUT_SORT_IDS = np.argsort(PATCH_ID_OUTPUT)


def _prepare_trigger_input(_a):
    A, B = 3, 192
    cut = 144
    _a = _a.reshape(-1, A)
    _a = _a.reshape(-1, A, B)
    _a = _a[..., :cut]
    _a = _a.reshape(_a.shape[0], -1)
    _a = _a.T
    _a = _a[PATCH_ID_INPUT_SORT_IDS]
    return _a


def _prepare_trigger_output(_a):
    A, B, C = 3, 18, 8

    _a = np.unpackbits(_a.reshape(-1, A, B, 1), axis=-1)
    _a = _a[..., ::-1]
    _a = _a.reshape(-1, A*B*C).T
    return _a[PATCH_ID_OUTPUT_SORT_IDS]
