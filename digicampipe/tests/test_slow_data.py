import os
import warnings

import numpy as np
from cts_core.camera import Camera
from pkg_resources import resource_filename

from digicampipe.instrument import geometry
from digicampipe.io.event_stream import event_stream, add_slow_data, \
    calibration_event_stream, add_slow_data_calibration

warnings.simplefilter("ignore")

example_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)

digicam_config_file = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'camera_config.cfg'
    )
)

aux_basepath = resource_filename('digicampipe', 'tests/resources/')

digicam = Camera(_config_file=digicam_config_file)
digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)


def test_add_slow_data():
    data_stream = event_stream(example_file_path, max_events=100)
    data_stream = add_slow_data(data_stream, basepath=aux_basepath)
    ts_slow = []
    ts_data = []
    for event in data_stream:
        tel_id = event.r0.tels_with_data[0]
        ts_slow.append(event.slow_data.DriveSystem.timestamp * 1e-3)
        ts_data.append(event.r0.tel[tel_id].local_camera_clock * 1e-9)
    ts_slow = np.array(ts_slow)
    ts_data = np.array(ts_data)
    diff = ts_data - ts_slow
    assert (diff <= 1.1).all()


def test_add_slow_data_calibration():
    data_stream = calibration_event_stream(example_file_path, max_events=100)
    data_stream = add_slow_data_calibration(data_stream, basepath=aux_basepath)
    ts_slow = []
    ts_data = []
    for event in data_stream:
        ts_slow.append(event.slow_data.DriveSystem.timestamp * 1e-3)
        ts_data.append(event.data.local_time * 1e-9)
    ts_slow = np.array(ts_slow)
    ts_data = np.array(ts_data)
    diff = ts_data - ts_slow
    assert (diff <= 1.1).all()


if __name__ == '__main__':
    test_add_slow_data_calibration()
    test_add_slow_data()