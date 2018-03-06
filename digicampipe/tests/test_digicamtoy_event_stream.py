from digicampipe.io.hdf5 import digicamtoy_event_source
from digicampipe.io.event_stream import event_stream
import pkg_resources
import os
import pytest

example_file_paths = []

for i in range(12):

    path = pkg_resources.resource_filename(
        'digicampipe',
        os.path.join(
            'tests',
            'resources',
            'digicamtoy',
            'test_digicamtoy_{}.hdf5'.format(i)
        )
    )

    example_file_paths.append(path)

TEL_WITH_DATA = 1
N_PIXELS = 1296


def test_event_source():

    for _ in digicamtoy_event_source(example_file_paths[0]):
        pass


def test_event_stream():

    for _ in event_stream(example_file_paths):

        pass


def test_n_pixels():

    for event in event_stream(example_file_paths, mc=True):

        assert event.inst.num_pixels[TEL_WITH_DATA] == N_PIXELS


def test_tel_with_data():

    for event in event_stream(example_file_paths, mc=True):

        assert TEL_WITH_DATA in event.r0.tels_with_data