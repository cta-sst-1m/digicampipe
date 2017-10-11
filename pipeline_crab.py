from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import  geometry
from cts_core.camera import Camera
from digicampipe.io.save_hillas import save_hillas_parameters
from digicamviewer.viewer import EventViewer2
from digicampipe.utils import utils
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

if __name__ == '__main__':

    #########################
    ##### CONFIGURATION #####
    #########################

    # Input configuration

    directory = '/sst1m/raw/2017/09/28/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in range(19, 23)]
    digicam_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    max_events = 10

    # Source coordinates
    source_x = 0. * u.mm
    source_y = 0. * u.mm

    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam, source_x=source_x, source_y=source_y)

    # Camera config file
    # File for pedestal calculation
    dark_baseline = np.load(directory + 'dark.npz')

    # Config for Hillas parameters analysis
    hillas_filename = directory + 'hillas_test.npz'
    n_showers = 100
    reclean = True

    # Noisy patch that triggered
    unwanted_patch = None  # [306, 318, 330, 342] #[391, 392, 403, 404, 405, 416, 417]

    # Noisy pixels not taken into account in Hillas
    pixel_not_wanted = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    additional_mask = np.ones(1296)
    additional_mask[pixel_not_wanted] = 0
    additional_mask = additional_mask > 0

    # Config for NSB evaluation
    n_bins = 1000

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

    peak_position = utils.fake_timing_hist(time_integration_options['n_samples'],
                                           time_integration_options['timing_width'],
                                           time_integration_options['central_sample'])

    time_integration_options['peak'], time_integration_options['mask'], time_integration_options['mask_edges'] = \
        utils.generate_timing_mask(time_integration_options['window_start'], time_integration_options['window_width'],
                                   peak_position)

    # Image cleaning configuration
    picture_threshold = 40
    boundary_threshold = 10
    shower_distance = 200 * u.mm

    # Filering on big showers
    min_photon = 100

    ####################
    ##### ANALYSIS #####
    ####################

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    data_stream = filter.set_pixels_to_zero(data_stream, unwanted_pixels=pixel_not_wanted)

    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.fill_flag(data_stream, unwanted_patch=unwanted_patch)

    # Fill the baseline (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    data_stream = filter.filter_event_types(data_stream, flags=[1])
    data_stream = filter.filter_missing_baseline(data_stream)

    # Run the r1 calibration (i.e baseline substraction)
    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)

    # Run the dl0 calibration (data reduction, does nothing)
    data_stream = dl0.calibrate_to_dl0(data_stream)

    # Run the dl1 calibration (compute charge in photons)
    # data_stream = dl1.calibrate_to_dl1(data_stream, time_integration_options, additional_mask=additional_mask, cleaning_threshold=6)
    data_stream = dl1.calibrate_to_dl1_better_cleaning(data_stream, time_integration_options,
                                                       additional_mask=additional_mask,
                                                       picture_threshold=picture_threshold,
                                                       boundary_threshold=boundary_threshold)

    data_stream = filter.filter_shower(data_stream, min_photon=min_photon)

    # Run the dl2 calibration (Hillas + classification + energy + direction)
    data_stream = dl2.calibrate_to_dl2(data_stream, reclean=reclean, shower_distance=shower_distance)
    # Save the hillas parameters
    # save_hillas_parameters(data_stream=data_stream, n_showers=n_showers, output_filename=hillas_filename)

    with plt.style.context('ggplot'):
        display = EventViewer2(data_stream, n_samples=50, camera_config_file=digicam_config_file, scale='lin')#, limits_colormap=[10, 500])
        display.draw()
        plt.show()
