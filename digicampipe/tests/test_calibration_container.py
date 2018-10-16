import os

import pkg_resources

from digicampipe.io.event_stream import calibration_event_stream, event_stream

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)


def test_calibration_event_stream():
    max_events = 10
    calib_stream = calibration_event_stream(example_file_path,
                                            max_events=max_events)

    values = []

    for event_calib in calib_stream:
        values.append([
            event_calib.data.adc_samples,
            event_calib.data.digicam_baseline,
            event_calib.pixel_id
        ])
    assert len(values) == max_events

    del calib_stream

    obs_stream = event_stream(example_file_path, max_events=max_events)

    for i, event in enumerate(obs_stream):
        event = list(event.r0.tel.values())[0]

        assert (values[i][0] == event.adc_samples).all()

        assert (values[i][1] == event.digicam_baseline).all()

        assert len(values[i][2]) == event.adc_samples.shape[0]


def test_event_type_enum_behavior():
    for event in calibration_event_stream(example_file_path):
        assert event.event_type in [event.event_type.PATCH7,
                                    event.event_type.INTERNAL]
