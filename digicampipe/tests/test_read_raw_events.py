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


def test_rawreader_can_work_with_absolute_path():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    rawzfitsreader.open(example_file_path + ':Events')
    raw = rawzfitsreader.readEvent()
    assert rawzfitsreader.getNumRows() == 10

    event = L0_pb2.CameraEvent()
    event.ParseFromString(raw)


def test_can_iterate_over_events():
    from digicampipe.io.protozfitsreader import ZFile

    zfits = ZFile(example_file_path)
    for __ in zfits.move_to_next_event():
        pass


def test_iteration_yield_expected_fields():
    from digicampipe.io.protozfitsreader import ZFile

    zfits = ZFile(example_file_path)

    for __ in zfits.move_to_next_event():
        # we just want to see, that the zfits file has all these
        # fields and we can access them
        zfits.get_event_number()
        zfits.event.telescopeID
        zfits.event.num_gains
        zfits.get_number_of_pixels()
        zfits.event.eventNumber
        zfits.get_pixel_flags()
        zfits.get_local_time()
        zfits.get_central_event_gps_time()
        zfits.get_camera_event_type()
        zfits.get_array_event_type()
        zfits.get_num_samples()
        zfits.get_adcs_samples()

        # expert mode fields
        zfits.get_trigger_input_traces()
        zfits.get_trigger_output_patch7()
        zfits.get_trigger_output_patch19()
        zfits.get_baseline()
