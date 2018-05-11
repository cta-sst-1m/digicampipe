# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.
This requires the protozfitsreader python library to be installed
"""
import logging
from tqdm import tqdm
from digicampipe.io.containers import DataContainer
import digicampipe.utils as utils
from protozfitsreader import ZFile
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
    loaded_telescopes = []
    event_stream = ZFile(url)

    if max_events is None:

        n_events = event_stream.numrows
    else:

        n_events = max_events

    for event_counter, event in tqdm(enumerate(event_stream), total=n_events,
                                     desc='Events', leave=False):
        if max_events is not None and event_counter > max_events:
            break

        data.r0.event_id = event_counter
        data.r0.tels_with_data = [event.telescope_id, ]

        # remove forbidden telescopes
        if allowed_tels:
            data.r0.tels_with_data = [
                list(filter(lambda x: x in data.r0.tels_with_data, sublist))
                for sublist in allowed_tels
            ]

        for tel_id in data.r0.tels_with_data:

            if tel_id not in loaded_telescopes:
                data.inst.num_channels[tel_id] = event.num_channels
                data.inst.num_pixels[tel_id] = event.n_pixels
                data.inst.geom[tel_id] = camera.geometry
                data.inst.cluster_matrix_7[tel_id] = camera.cluster_7_matrix
                data.inst.cluster_matrix_19[tel_id] = camera.cluster_19_matrix
                data.inst.patch_matrix[tel_id] = camera.patch_matrix
                data.inst.num_samples[tel_id] = event.num_samples
                loaded_telescopes.append(tel_id)

            r0 = data.r0.tel[tel_id]
            r0.camera_event_number = event.event_number
            r0.pixel_flags = event.pixel_flags
            r0.local_camera_clock = event.local_time
            r0.gps_time = event.central_event_gps_time
            r0.camera_event_type = event.camera_event_type
            r0.array_event_type = event.array_event_type
            r0.adc_samples = event.adc_samples

            r0.trigger_input_traces = event.trigger_input_traces
            r0.trigger_output_patch7 = event.trigger_output_patch7
            r0.trigger_output_patch19 = event.trigger_output_patch19
            r0.digicam_baseline = event.baseline

        yield data

    return


def count_number_events(file_list):
    """return sum of events in all files
    It is useful for fixed size array creation.

    :param file_list: list of zfits files
    :return: n_events: int
    """
    n_events = 0

    for filename in file_list:

        zfits = ZFile(filename)
        n_events += zfits.numrows

    return n_events
