import os
import warnings
import numpy as np
from pkg_resources import resource_filename

from digicampipe.io.event_stream import event_stream, add_slow_data, \
    calibration_event_stream, add_slow_data_calibration

warnings.simplefilter("ignore")

example_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'run150_100_evts.fits.fz'
    )
)

aux_basepath = resource_filename('digicampipe', 'tests/resources/')


def test_add_slow_data():
    data_stream = event_stream(example_file_path, max_events=100)
    data_stream = add_slow_data(
        data_stream, basepath=aux_basepath,
        aux_services=(
            'DigicamSlowControl',
            'MasterSST1M',
            'PDPSlowControl',
            'SafetyPLC',
            'DriveSystem',
        )
    )
    ts_digicam = []
    ts_master = []
    ts_pdp = []
    ts_safety = []
    ts_drive = []
    ts_data = []
    for event in data_stream:
        tel_id = event.r0.tels_with_data[0]
        ts_digicam.append(event.slow_data.DigicamSlowControl.timestamp * 1e-3)
        ts_master.append(event.slow_data.MasterSST1M.timestamp * 1e-3)
        ts_pdp.append(event.slow_data.PDPSlowControl.timestamp * 1e-3)
        ts_safety.append(event.slow_data.SafetyPLC.timestamp * 1e-3)
        ts_drive.append(event.slow_data.DriveSystem.timestamp * 1e-3)
        ts_data.append(event.r0.tel[tel_id].local_camera_clock * 1e-9)
    ts_digicam = np.array(ts_digicam)
    ts_master = np.array(ts_master)
    ts_pdp = np.array(ts_pdp)
    ts_safety = np.array(ts_safety)
    ts_drive = np.array(ts_drive)
    ts_data = np.array(ts_data)
    assert ((ts_data - ts_digicam) <= 1.1).all()
    assert ((ts_data - ts_master) <= 1.1).all()
    assert ((ts_data - ts_pdp) <= 1.1).all()
    assert ((ts_data - ts_safety) <= 1.1).all()
    assert ((ts_data - ts_drive) <= 1.1).all()


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
