import pytest

import pkg_resources
import os
from os.path import relpath

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_10evts.fits.fz'
    )
)


@pytest.mark.skip(reason="we know the current version does not raise")
def test_zfile_raises_on_wrong_path():
    from digicampipe.io.protozfitsreader import ZFile
    with pytest.raises(FileNotFoundError):
        ZFile('foo.bar')


def test_zfile_opens_correct_path():
    from digicampipe.io.protozfitsreader import ZFile
    ZFile(example_file_path)


def test_rawreader_can_work_with_relative_path():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    relative_test_file_path = os.path.relpath(example_file_path)
    rawzfitsreader.open(relative_test_file_path + ':Events')
    raw = rawzfitsreader.readEvent()
    assert rawzfitsreader.getNumRows() == 10

    event = L0_pb2.CameraEvent()
    event.ParseFromString(raw)


def test_rawreader_cannot_work_with_absolute_path():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    rawzfitsreader.open(example_file_path + ':Events')
    raw = rawzfitsreader.readEvent()
    assert rawzfitsreader.getNumRows() == 10

    event = L0_pb2.CameraEvent()
    event.ParseFromString(raw)


def test_can_iterate_over_events():
    from digicampipe.io.protozfitsreader import ZFile
    print(example_file_path)
    zfits = ZFile(example_file_path)
    event_stream = zfits.move_to_next_event()
    for __ in event_stream:
        pass

