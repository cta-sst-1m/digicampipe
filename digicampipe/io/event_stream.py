from digicampipe.io import zfits, hdf5, hessio_digicam
from .auxservice import AuxService
from collections import namedtuple
from digicampipe.io.containers_calib import CalibrationContainer
from tqdm import tqdm
import numpy as np
import os


def event_stream(filelist, source=None, max_events=None, **kwargs):
    """
    Iterable of events in the form of `DataContainer`.

    Parameters
    ----------
    filelist : list-like of paths, or a single path(string).
    source: function-like or None
        the function to be used for reading the events.
        If not specified it is guessed.
        possible choices are:
            * digicampipe.io.zfits.zfits_event_source
            * digicampipe.io.hdf5.digicamtoy_event_source
            * digicampipe.io.hessio_digicam.hessio_event_source

    kwargs: parameters for event_source
        Some event_sources need special parameters to work, c.f. their doc.
    """

    # If the caller gives us a path and not a list of paths,
    # we convert it to a list.
    # This is not clean but convenient.
    if isinstance(filelist, (str, bytes)):
        filelist = [filelist]
    n_files = len(filelist)
    count = 0

    for file in filelist:

        if not os.path.exists(file):

            raise FileNotFoundError('File {} does not exists'.format(file))

    if max_events is None:

        max_events = np.inf

    if n_files == 1:

        file_stream = filelist

    else:

        file_stream = tqdm(filelist, total=n_files, desc='Files', leave=True)
    for file in file_stream:
        if source is None:
            source = guess_source_from_path(file)
        data_stream = source(url=file, **kwargs)
        try:
            for event in data_stream:
                if count >= max_events:
                    raise StopIteration
                count += 1
                yield event
        except EOFError as e:
            print('WARNING: unexpected end of file', file, ':', e)
        except SystemError as e:
            print('WARNING: system error.', e)


def calibration_event_stream(path,
                             pixel_id=[...],
                             max_events=None,
                             **kwargs):
    """
    Event stream for the calibration of the camera based on the observation
    event_stream()
    """
    pixel_id = tuple(pixel_id)
    container = CalibrationContainer()
    for event in event_stream(path, max_events=max_events, **kwargs):
        r0_event = list(event.r0.tel.values())[0]
        container.pixel_id = np.arange(r0_event.adc_samples.shape[0])[pixel_id]
        container.event_type = r0_event.camera_event_type
        container.data.adc_samples = r0_event.adc_samples[pixel_id]
        container.data.digicam_baseline = r0_event.digicam_baseline[pixel_id]
        container.data.local_time = r0_event.local_camera_clock
        container.data.gps_time = r0_event.gps_time
        container.event_id = r0_event.camera_event_number
        yield container


def guess_source_from_path(path):
    if path.endswith('.fits.fz'):
        return zfits.zfits_event_source
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        return hdf5.digicamtoy_event_source
    else:
        return hessio_digicam.hessio_event_source


def add_slow_data(
    data_stream,
    aux_services=[
        'DigicamSlowControl',
        'MasterSST1M',
        'PDPSlowControl',
        'SafetyPLC',
        'DriveSystem',
    ],
    basepath=None
):
    services = {
        name: AuxService(name, basepath)
        for name in aux_services
    }

    SlowDataContainer = namedtuple('SlowDataContainer', aux_services)
    for event_id, event in enumerate(data_stream):
        tel = event.r0.tels_with_data[0]
        services_event = {
            name: service.at(event.r0.tel[tel].local_camera_clock)
            for (name, service) in services.items()
        }
        event.slow_data = SlowDataContainer(**services_event)
        yield event
