import os
from astropy import units as u
import pkg_resources

from digicampipe.io.hessio import hessio_get_list_event_ids
from digicampipe.io.hessio import hessio_event_source
from digicampipe.io.event_stream import event_stream, calibration_event_stream

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'simtel',
        '1_triggered_events_10_TeV.simtel.gz'
    )
)

example_file_path_1 = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'simtel',
        'file-pedestal.simtel.gz'
    )
)


EVENT_ID = 1
EVENTS_IN_EXAMPLE_FILE = 1
ENERGY = 10 * u.TeV


def test_and_benchmark_event_source(benchmark):
    @benchmark
    def loop():
        for _ in hessio_event_source(example_file_path):
            pass


def test_count_number_event():
    events_id = hessio_get_list_event_ids(example_file_path)
    assert len(events_id) == EVENTS_IN_EXAMPLE_FILE


def test_event_id():
    for data in hessio_event_source(example_file_path):
        event_id = data.r0.event_id
        energy = data.mc.energy
        break
    assert event_id == EVENT_ID
    assert energy == ENERGY


def test_event_stream():
    events = event_stream([example_file_path])
    for event in events:
        event_id = event.r0.event_id
        energy = event.mc.energy
        break
    assert event_id == EVENT_ID
    assert energy == ENERGY


def test_event_stream_with_event_id_none():
    events = event_stream([example_file_path],
                          event_id=None)
    for _ in events:

        pass


def test_event_stream_pedestal_file():
    events = event_stream([example_file_path_1])
    for _ in events:

        pass


def test_calibration_event_stream():
    events = calibration_event_stream([example_file_path])
    for event in events:
        event_id = event.event_id
        energy = event.mc.energy
        break
    assert event_id == EVENT_ID
    assert energy == ENERGY


if __name__ == '__main__':
    test_count_number_event()
    test_event_id()
    test_event_stream()
    test_calibration_event_stream()
