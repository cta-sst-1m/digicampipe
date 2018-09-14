import os

import pkg_resources

from digicampipe.io.event_stream import event_stream
from digicampipe.io.hdf5 import digicamtoy_event_source

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'digicamtoy',
        'test_digicamtoy_0.hdf5'
    )
)

TEL_WITH_DATA = 1
N_PIXELS = 1296


def test_event_source():
    for _ in digicamtoy_event_source(example_file_path):
        pass


def test_event_source_speed_100_events(benchmark):
    @benchmark
    def func():
        for _, i in zip(event_stream(example_file_path), range(100)):
            pass
        assert i == 99


def test_event_stream():
    for _ in event_stream(example_file_path):
        pass


def test_n_pixels():
    for event in event_stream(example_file_path):
        assert event.inst.num_pixels[TEL_WITH_DATA] == N_PIXELS


def test_tel_with_data():
    for event in event_stream(example_file_path):
        assert TEL_WITH_DATA in event.r0.tels_with_data
