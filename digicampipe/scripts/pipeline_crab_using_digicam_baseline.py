from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_hillas import save_hillas_parameters_in_list
from digicampipe.visualization import EventViewer
from digicampipe.utils import utils
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


def main(
    files,
    baseline_path,
    min_photon=20,
    display=False,
):
    hillas_parameters = []

    # Input/Output files
    dark_baseline = np.load(baseline_path)

    # Config for NSB + baseline evaluation
    n_bins = 1000

    # Config for Hillas parameters analysis
    reclean = True

    # Noisy pixels not taken into account in Hillas
    pixel_not_wanted = [
        1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    additional_mask = np.ones(1296)
    additional_mask[pixel_not_wanted] = 0
    additional_mask = additional_mask > 0

    # Integration configuration (signal reco.)
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

    # Image cleaning configuration
    picture_threshold = 15
    boundary_threshold = 10
    shower_distance = 200 * u.mm

    # Define the event stream
    data_stream = event_stream(files)
    # Clean pixels
    data_stream = filter.set_pixels_to_zero(
        data_stream, unwanted_pixels=pixel_not_wanted)
    # Compute baseline with clocked triggered events (sliding average over n_bins)
    data_stream = random_triggers.fill_baseline_r0_but_not_baseline(data_stream, n_bins=n_bins)

    # Stop events that are not triggered by DigiCam algorithm (end of clocked triggered events)
    data_stream = filter.filter_event_types(data_stream, flags=[1, 2])

    # Run the r1 calibration (i.e baseline substraction)
    data_stream = r1.calibrate_to_r1_using_digicam_baseline(data_stream, dark_baseline)
    # Run the dl0 calibration (data reduction, does nothing)
    data_stream = dl0.calibrate_to_dl0(data_stream)
    # Run the dl1 calibration (compute charge in photons + cleaning)
    data_stream = dl1.calibrate_to_dl1(data_stream,
                                       time_integration_options,
                                       additional_mask=additional_mask,
                                       picture_threshold=picture_threshold,
                                       boundary_threshold=boundary_threshold)
    # Return only showers with total number of p.e. above min_photon
    data_stream = filter.filter_shower(
        data_stream, min_photon)
    # Run the dl2 calibration (Hillas)
    data_stream = dl2.calibrate_to_dl2(
        data_stream, reclean=reclean, shower_distance=shower_distance)

    if display:

        with plt.style.context('ggplot'):
            display = EventViewer(data_stream)
            display.draw()
    else:
        save_hillas_parameters_in_list(
            data_stream=data_stream,
            list=hillas_parameters)

    for _ in data_stream:
        pass

    return hillas_parameters
