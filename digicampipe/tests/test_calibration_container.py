from digicampipe.io.event_stream import calibration_event_stream, event_stream
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


def test_calibration_event_stream():

    max_events = 10
    telescope_id = 1

    calib_stream = calibration_event_stream(example_file_path,
                                            telescope_id,
                                            max_events)

    obs_stream = event_stream(example_file_path, max_events=max_events)

    values = []

    for event_calib in calib_stream:

        values.append([
            event_calib.data.adc_samples,
            event_calib.data.digicam_baseline,
            event_calib.n_pixels
            ])

    for i, event in enumerate(obs_stream):

        event = event.r0.tel[telescope_id]

        assert (values[i][0] == event.adc_samples).all()

        assert (values[i][1] == event.digicam_baseline).all()

        assert values[i][2] == event.adc_samples.shape[0]

