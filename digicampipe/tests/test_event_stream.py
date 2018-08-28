from digicampipe.io.event_stream import event_stream
import pkg_resources
import os

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'SST1M_01_20180822_001.fits.fz'
    )
)


def test_event_source_new_style():

    for _ in event_stream(example_file_path):
        pass

