import os

import pkg_resources

from digicampipe.io.event_stream import event_stream

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


def test_event_type_enum_behavior():
    for event in event_stream(example_file_path):
        for _, r0 in event.r0.tel.items():
            assert r0.camera_event_type.INTRNL in r0.camera_event_type
