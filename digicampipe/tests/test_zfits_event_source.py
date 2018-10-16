import os

import pkg_resources

from digicampipe.io.event_stream import event_stream
from digicampipe.io.zfits import count_number_events
from digicampipe.io.zfits import zfits_event_source

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)

FIRST_EVENT_ID = 97750287
LAST_EVENT_ID = 97750386
EVENTS_IN_EXAMPLE_FILE = 100


def test_and_benchmark_event_source(benchmark):
    @benchmark
    def loop():
        for _ in zfits_event_source(example_file_path):
            pass


def test_count_number_event():
    n_files = 10
    files = [example_file_path] * n_files  # create a list of files

    assert count_number_events(files) == n_files * EVENTS_IN_EXAMPLE_FILE


def test_event_id():
    event_id = LAST_EVENT_ID - 3

    for data in zfits_event_source(example_file_path,
                                   event_id=event_id):
        tel_id = 1
        r0 = data.r0.tel[tel_id]
        number = r0.camera_event_number

        break

    assert number == event_id

    for data in event_stream(example_file_path, event_id=event_id):
        tel_id = 1
        r0 = data.r0.tel[tel_id]
        number = r0.camera_event_number

        break

    assert number == event_id


if __name__ == '__main__':
    test_event_id()
