from pkg_resources import resource_filename
from glob import glob
import os
import warnings
import numpy as np

from cts_core.camera import Camera
from digicampipe.utils import geometry
from digicampipe.io.event_stream import event_stream, add_slow_data

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

aux_basepath = resource_filename('digicampipe', 'tests/resources/slow/')


digicam = Camera(_config_file=digicam_config_file)
digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)


def test_add_slow_data():
    data_stream = event_stream(
        file_list=[example_file_path],
        camera_geometry=digicam_geometry,
        camera=digicam,
        max_events=100
    )
    data_stream = add_slow_data(data_stream, basepath=aux_basepath)
    for event in data_stream:
        pass
    print(event.slow_data.__dict__.keys())
    print(event.slow_data.DriveSystem.dtype)
