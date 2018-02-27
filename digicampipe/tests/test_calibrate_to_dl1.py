import os.path
import pkg_resources
import numpy as np
from tempfile import TemporaryDirectory

from digicampipe.calib.camera import filter, r1, random_triggers, dl0
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


def make_dark_base_line(dir):
    from digicampipe.calib.camera import filter
    from digicampipe.io.event_stream import event_stream
    from digicampipe.io.save_adc import save_dark

    outfile_path = os.path.join(dir, "dark_baseline.npz")

    data_stream = event_stream(example_file_path)
    data_stream = filter.set_pixels_to_zero(data_stream)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = save_dark(data_stream, outfile_path)
    for _ in data_stream:
        pass

    return outfile_path


def test_calibrate_to_dl1():
    from digicampipe.calib.camera.dl1 import calibrate_to_dl1

    with TemporaryDirectory() as temp_dir:
        dark_baseline = make_dark_base_line(temp_dir)

        n_bins = 1000
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
        data_stream = dl0.calibrate_to_dl0(data_stream)
        data_stream = calibrate_to_dl1(
            data_stream,
            time_integration_options,
            additional_mask=additional_mask,
            picture_threshold=15,
            boundary_threshold=10,
        )
