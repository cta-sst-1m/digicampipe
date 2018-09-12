import os.path
from tempfile import TemporaryDirectory

import numpy as np
import pkg_resources

from digicampipe.calib import filter
from digicampipe.calib.camera import r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import utils

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)


def make_dark_base_line():
    from digicampipe.calib.camera import filter
    from digicampipe.io.event_stream import event_stream
    from digicampipe.io.save_adc import save_dark

    with TemporaryDirectory() as temp_dir:
        outfile_path = os.path.join(temp_dir, "dark_baseline.npz")

        data_stream = event_stream(example_file_path)
        data_stream = filter.set_pixels_to_zero(
            data_stream,
            unwanted_pixels=[],
        )
        data_stream = filter.filter_event_types(data_stream, flags=[8])
        data_stream = save_dark(data_stream, outfile_path)
        for event_counter, _ in enumerate(data_stream):
            pass
        assert event_counter >= 5

        npz_file = np.load(outfile_path)
        assert not np.any(np.isnan(npz_file['baseline']))
        assert not np.any(np.isnan(npz_file['standard_deviation']))
        assert len(npz_file['baseline']) == 1296

    return npz_file


def test_calibrate_to_dl1():
    from digicampipe.calib.camera.dl1 import calibrate_to_dl1

    # The next 50 lines are just setp.
    dark_baseline = make_dark_base_line()

    n_bins = 50
    pixel_not_wanted = [
        1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    additional_mask = np.ones(1296)
    additional_mask[pixel_not_wanted] = 0
    additional_mask = additional_mask > 0

    time_integration_options = {'mask': None,
                                'mask_edges': None,
                                'peak': None,
                                'window_start': 3,
                                'window_width': 7,
                                'threshold_saturation': np.inf,
                                'n_samples': 50,
                                'timing_width': 6,
                                'central_sample': 11}

    peak_position = utils.fake_timing_hist(
        time_integration_options['n_samples'],
        time_integration_options['timing_width'],
        time_integration_options['central_sample'])

    (
        time_integration_options['peak'],
        time_integration_options['mask'],
        time_integration_options['mask_edges']
    ) = utils.generate_timing_mask(
        time_integration_options['window_start'],
        time_integration_options['window_width'],
        peak_position
    )

    data_stream = event_stream(example_file_path)
    data_stream = filter.set_pixels_to_zero(
        data_stream,
        unwanted_pixels=pixel_not_wanted,
    )
    data_stream = random_triggers.fill_baseline_r0(
        data_stream,
        n_bins=n_bins,
    )
    data_stream = filter.filter_event_types(data_stream, flags=[1, 2])
    data_stream = filter.filter_missing_baseline(data_stream)

    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)

    # This is the function under test
    data_stream = calibrate_to_dl1(
        data_stream,
        time_integration_options,
        additional_mask=additional_mask,
        picture_threshold=15,
        boundary_threshold=10,
    )

    # This is the actual test:
    for event_counter, event in enumerate(data_stream):
        for dl1 in event.dl1.tel.values():

            assert dl1.cleaning_mask.shape == (1296, )
            assert dl1.cleaning_mask.dtype == np.bool

            assert not np.isnan(dl1.time_spread)

            assert dl1.pe_samples.shape == (1296, )
            assert dl1.pe_samples.dtype == np.float64

            assert dl1.on_border.shape == ()
            assert dl1.on_border.dtype == np.bool

    assert event_counter >= 86
