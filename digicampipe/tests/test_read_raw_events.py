import pytest

import pkg_resources
import os
from os.path import relpath
import numpy as np
import warnings

warnings.simplefilter("ignore")

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_10evts.fits.fz'
    )
)

FIRST_EVENT_IN_EXAMPLE_FILE = 97750287
TELESCOPE_ID_IN_EXAMPLE_FILE = 1
EVENTS_IN_EXAMPLE_FILE = 10
EXPECTED_LOCAL_TIME = [
    1.5094154944067896e+18,
    1.509415494408104e+18,
    1.509415494408684e+18,
    1.509415494415717e+18,
    1.5094154944180828e+18,
    1.5094154944218719e+18,
    1.5094154944245553e+18,
    1.5094154944267853e+18,
    1.509415494438982e+18,
    1.5094154944452902e+18
]
EXPECTED_GPS_TIME = [0] * EVENTS_IN_EXAMPLE_FILE

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
    assert rawzfitsreader.getNumRows() == EVENTS_IN_EXAMPLE_FILE

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


def test_rawreader_can_work_with_absolute_path():
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import L0_pb2

    rawzfitsreader.open(example_file_path + ':Events')
    raw = rawzfitsreader.readEvent()
    assert rawzfitsreader.getNumRows() == EVENTS_IN_EXAMPLE_FILE

    event = L0_pb2.CameraEvent()
    event.ParseFromString(raw)


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
        event.num_gains
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


def test_event_number():
    from digicampipe.io.protozfitsreader import ZFile

    event_numbers = [
        event.event_number
        for event in ZFile(example_file_path)
    ]
    expected_event_numbers = [
        FIRST_EVENT_IN_EXAMPLE_FILE + i
        for i in range(EVENTS_IN_EXAMPLE_FILE)
    ]
    assert event_numbers == expected_event_numbers


def test_telescope_ids():
    from digicampipe.io.protozfitsreader import ZFile
    telescope_ids = [
        event.telescope_id
        for event in ZFile(example_file_path)
    ]
    expected_ids = [TELESCOPE_ID_IN_EXAMPLE_FILE] * EVENTS_IN_EXAMPLE_FILE
    assert telescope_ids == expected_ids

@pytest.mark.xfail
def test_num_gains():
    from digicampipe.io.protozfitsreader import ZFile
    num_gains = [
        event.num_gains
        for event in ZFile(example_file_path)
    ]
    expected_num_gains = [0] * EVENTS_IN_EXAMPLE_FILE
    assert num_gains == expected_num_gains

def test_n_pixel():
    from digicampipe.io.protozfitsreader import ZFile
    n_pixel = [
        event.n_pixels
        for event in ZFile(example_file_path)
    ]
    assert n_pixel == [1296] * EVENTS_IN_EXAMPLE_FILE


def test_pixel_flags():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    pixel_flags = [
        zfits.get_pixel_flags()
        for __ in zfits.move_to_next_event()
    ]
    expected_pixel_flags = [
        np.ones(1296, dtype=np.bool)
    ] * EVENTS_IN_EXAMPLE_FILE

    for actual, expected in zip(pixel_flags, expected_pixel_flags):
        assert (actual == expected).all()

def test_local_time():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    local_time = [
        zfits.get_local_time()
        for __ in zfits.move_to_next_event()
    ]
    assert local_time == EXPECTED_LOCAL_TIME

def test_gps_time():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    gps_time = [
        zfits.get_central_event_gps_time()
        for __ in zfits.move_to_next_event()
    ]
    assert gps_time == EXPECTED_GPS_TIME


def test_camera_event_type():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    camera_event_type = [
        zfits.get_camera_event_type()
        for __ in zfits.move_to_next_event()
    ]
    assert camera_event_type == [1, 1, 1, 1, 1, 8, 1, 1, 1, 1]


def test_array_event_type():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    array_event_type = [
        zfits.get_array_event_type()
        for __ in zfits.move_to_next_event()
    ]
    assert array_event_type == [0] * EVENTS_IN_EXAMPLE_FILE


def test_num_samples():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    num_samples = [
        zfits.get_num_samples()
        for __ in zfits.move_to_next_event()
    ]
    assert num_samples == [50] * EVENTS_IN_EXAMPLE_FILE


def test_adc_samples():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    adc_samples = [
        zfits.get_adcs_samples()
        for __ in zfits.move_to_next_event()
    ]

    for actual in adc_samples:
        assert actual.dtype == np.int16
        assert actual.shape == (1296, 50)

    adc_samples = np.array(adc_samples)

    # these are 12 bit ADC values, so the range must
    # can at least be asserted
    assert adc_samples.min() == 0
    assert adc_samples.max() == (2**12) - 1


def test_trigger_input_traces():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    trigger_input_traces = [
        zfits.get_trigger_input_traces()
        for __ in zfits.move_to_next_event()
    ]

    for actual in trigger_input_traces:
        assert actual.dtype == np.uint8
        assert actual.shape == (432, 50)


def test_trigger_output_patch7():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    trigger_output_patch7 = [
        zfits.get_trigger_output_patch7()
        for __ in zfits.move_to_next_event()
    ]

    for actual in trigger_output_patch7:
        assert actual.dtype == np.uint8
        assert actual.shape == (432, 50)


def test_trigger_output_patch19():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    trigger_output_patch19 = [
        zfits.get_trigger_output_patch19()
        for __ in zfits.move_to_next_event()
    ]

    for actual in trigger_output_patch19:
        assert actual.dtype == np.uint8
        assert actual.shape == (432, 50)

def test_baseline():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    baseline = [
        zfits.get_baseline()
        for __ in zfits.move_to_next_event()
    ]

    for actual in baseline:
        assert actual.dtype == np.int16
        assert actual.shape == (1296,)

    baseline = np.array(baseline)

    baseline_deviation_between_events = baseline.std(axis=0)
    # I don't know if this is a good test, but I assume baseline should
    # not vary too much between events, so I had a look at these.
    assert baseline_deviation_between_events.max() < 60
    assert baseline_deviation_between_events.mean() < 2
