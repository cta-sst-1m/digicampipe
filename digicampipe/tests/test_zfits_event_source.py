from digicampipe.io.zfits import zfits_event_source
import pkg_resources
import os
import warnings

from cts_core.camera import Camera
from digicampipe.utils import geometry


warnings.simplefilter("ignore")

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


def test_and_benchmark_event_source(benchmark):

    @benchmark
    def loop():
        for _ in zfits_event_source(example_file_path):
            pass


def test_count_number_event():

    from digicampipe.io.zfits import count_number_events
    EVENTS_IN_EXAMPLE_FILE = 100

    n_files = 10
    files = [example_file_path] * n_files  # create a list of files

    assert count_number_events(files) == n_files * EVENTS_IN_EXAMPLE_FILE
