from digicampipe.io.event_stream import event_stream
import pkg_resources
import os

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)


def test_event_source_new_style():

    for _ in event_stream(example_file_path):
        pass


def test_event_source_speed_100_events(benchmark):

    @benchmark
    def func():
        for _, i in zip(event_stream(example_file_path), range(100)):

            pass

        assert i == 99
