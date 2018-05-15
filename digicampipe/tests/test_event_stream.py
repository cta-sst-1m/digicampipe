from digicampipe.io.event_stream import event_stream
import pkg_resources
import os

from cts_core.camera import Camera
from digicampipe.utils import geometry


example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)

digicam_config_file = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'camera_config.cfg'
    )
)

digicam = Camera(_config_file=digicam_config_file)
digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)


def test_event_source_new_style():

    for _ in event_stream(example_file_path):
        pass


def test_event_source_speed_100_events(benchmark):

    @benchmark
    def func():
        for _, i in zip(event_stream(example_file_path), range(100)):

            pass

        assert i == 99
