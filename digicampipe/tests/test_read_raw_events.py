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

def test_rawreader_can_work_with_relative_path():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    relative_test_file_path = os.path.relpath(example_file_path)
    rawzfitsreader.open(relative_test_file_path + ':Events')
    raw = rawzfitsreader.readEvent()
    assert rawzfitsreader.getNumRows() == 10

    event = L0_pb2.CameraEvent()
    event.ParseFromString(raw)


def test_rawreader_can_work_with_absolute_path():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    rawzfitsreader.open(example_file_path + ':Events')
    raw = rawzfitsreader.readEvent()
    assert rawzfitsreader.getNumRows() == 10

    event = L0_pb2.CameraEvent()
    event.ParseFromString(raw)

@pytest.mark.xfail
def test_rawreader_can_read_runheader():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    rawzfitsreader.open(example_file_path + ':RunHeader')

    raw = rawzfitsreader.readEvent()
    assert len(raw) > 0

    header = L0_pb2.CameraRunHeader()
    header.ParseFromString(raw)


def test_zfile_raises_on_wrong_path():
    from digicampipe.io.protozfitsreader import ZFile
    with pytest.raises(FileNotFoundError):
        ZFile('foo.bar')


def test_zfile_opens_correct_path():
    from digicampipe.io.protozfitsreader import ZFile
    ZFile(example_file_path)


def test_can_iterate_over_events():
    from digicampipe.io.protozfitsreader import ZFile

    for __ in ZFile(example_file_path):
        pass


def test_iteration_yield_expected_fields():
    from digicampipe.io.protozfitsreader import ZFile

    for event in ZFile(example_file_path):
        # we just want to see, that the zfits file has all these
        # fields and we can access them
        event.event_id
        event.telescope_id
        event.num_channels
        event.n_pixels
        event.event_number
        event.pixel_flags

        event.local_time
        event.central_event_gps_time
        event.camera_event_type
        event.array_event_type
        event.num_samples
        event.adc_samples

        # expert mode fields
        event.trigger_input_traces
        event.trigger_output_patch7
        event.trigger_output_patch19
        event.baseline
