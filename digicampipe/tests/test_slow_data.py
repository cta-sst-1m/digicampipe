from pkg_resources import resource_filename
from glob import glob
import os
import warnings
import numpy as np

from cts_core.camera import Camera
from digicampipe.utils import geometry
from digicampipe.io.event_stream import event_stream, add_slow_data, \
    get_slow_data_info, get_slow_event

warnings.simplefilter("ignore")

example_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'slow',
        'SST1M01_0_000.006.fits.fz'
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

slow_file_list = glob(
    resource_filename('digicampipe', 'tests/resources/slow/*.fits')
)

digicam = Camera(_config_file=digicam_config_file)
digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)


SLOW_CLASSES = ["DigicamSlowControl", "DriveSystem", "PDPSlowControl",
                "SafetyPLC", "MasterSST1M"]


def test_get_slow_data_info():
    slow_info_structs = get_slow_data_info(slow_file_list)
    for class_name in SLOW_CLASSES:
        assert(class_name in slow_info_structs.keys())
        assert(len(slow_info_structs[class_name])>0)
        data_struct = slow_info_structs[class_name][0]
        assert('ts_min' in data_struct.keys())
        assert('ts_max' in data_struct.keys())
        assert('hdu' in data_struct.keys())


def test_get_slow_event():
    slow_info_structs = get_slow_data_info(slow_file_list)
    for class_name in SLOW_CLASSES:
        ts_min_all = [info_file['ts_min']
                      for info_file in slow_info_structs[class_name]]
        ts_max_all = [info_file['ts_max']
                      for info_file in slow_info_structs[class_name]]
        ts_min_all = np.array(ts_min_all)
        ts_max_all = np.array(ts_max_all)
        for file in range(len(slow_info_structs[class_name])):
            info_struct = slow_info_structs[class_name][file]
            ts_min = info_struct['ts_min']
            ts_max = info_struct['ts_max']
            if ts_min == ts_max:
                continue
            data_ts_ok = (ts_max + ts_min) / 2
            e, h = get_slow_event(slow_info_structs[class_name], data_ts_ok)
            assert(e is not None)
            assert(h == info_struct['hdu'])
        data_ts_toobig = np.max(ts_max_all) + 1
        e, h = get_slow_event(slow_info_structs[class_name], data_ts_toobig)
        assert(e is None)
        assert(h is None)
        data_ts_toosmall = np.min(ts_min_all) - 1
        e, h = get_slow_event(slow_info_structs[class_name], data_ts_toosmall)
        assert(e is None)
        assert(h is None)


def test_add_slow_data():
    data_stream = event_stream(
        file_list=[example_file_path],
        # expert_mode=True,
        camera_geometry=digicam_geometry,
        camera=digicam,
        max_events=100
    )
    data_stream = add_slow_data(data_stream, slow_file_list=slow_file_list)
    ts_slow = []
    ts_data = []
    diff = []
    i = 0
    for event in data_stream:
        if len(event.r0.tels_with_data) == 0:
            print('WARNING in test_add_slow_data():',
                  'event does not have a telescope with r0 data.')
        tel = event.r0.tels_with_data[0]
        if not np.isreal(event.slow_data.drivesystem.timestamp):
            # print("WARNING: could not find timestamp for event")
            continue
        ts_slow.append(event.slow_data.drivesystem.timestamp * 1e-3)
        ts_data.append(event.r0.tel[tel].local_camera_clock * 1e-9)
        diff.append(ts_data[-1] - ts_slow[-1])
        i += 1
    assert(i > 0)
    assert(all(np.array(diff) <= 1.1))
    return ts_data, ts_slow, diff


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    test_get_slow_data_info()
    test_get_slow_event()
    ts_data, ts_slow, diff = test_add_slow_data()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ts_slow-np.min(ts_data), ts_data-np.min(ts_data), '.')
    plt.xlabel('ts(slow data) - ts_min, s')
    plt.ylabel('ts(digicam) - ts_min, s')
    plt.subplot(2, 1, 2)
    plt.plot(diff, '.')
    plt.xlabel('event')
    plt.ylabel('ts(digicam) - ts(slow data), s')
    plt.show()
