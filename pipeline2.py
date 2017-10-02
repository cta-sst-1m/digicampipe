from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import event_stream
from digicamviewer.viewer import EventViewer, EventViewer2
from digicampipe.utils import utils
import numpy as np
import matplotlib.pyplot as plt
from cts_core import camera as cam

if __name__ == '__main__':
    # Data configuration
    # directory = '/data/datasets/CTA/REALDATA/'
    # directory = '/home/alispach/blackmonkey/calib_data/first_light/20170831/'
    directory = '/home/alispach/data/CRAB_01/'
    # directory = '/home/alispach/Downloads/'

    filename = directory + 'CRAB_01_0_000.%03d_remapped.fits.fz'
    file_list = [filename % number for number in range(12, 13)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_baseline = np.load(directory + 'dark.npz')

    # Trigger configuration
    unwanted_patch = None  # [306, 318, 330, 342] #[391, 392, 403, 404, 405, 416, 417]

    pixel_not_wanted = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]

    additional_mask = np.ones(1296)
    additional_mask[pixel_not_wanted] = 0

    additional_mask = additional_mask > 0

    # print(patch_not_wanted)

    # Integration configuration
    time_integration_options = {'mask':None,
                                'mask_edges':None,
                                'peak':None,
                                'window_start':3,
                                'window_width':7,
                                'threshold_saturation':3500,
                                'n_samples':50,
                                'timing_width':6,
                                'central_sample':11}

    peak_position = utils.fake_timing_hist(time_integration_options['n_samples'], time_integration_options['timing_width'],
                                     time_integration_options['central_sample'])
    time_integration_options['peak'], time_integration_options['mask'], time_integration_options['mask_edges'] = \
        utils.generate_timing_mask(time_integration_options['window_start'],
                             time_integration_options['window_width'],
                             peak_position)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, geom_file=camera_config_file, remapped=False)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.fill_flag(data_stream, unwanted_patch=unwanted_patch)
    # Fill the baseline (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=1000)

    data_stream = filter.filter_flag(data_stream, flags=[1])
    data_stream = filter.filter_baseline_zero(data_stream)
    # Run the r1 calibration (i.e baseline substraction)
    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)
    # Run the dl0 calibration (data reduction, does nothing)
    data_stream = dl0.calibrate_to_dl0(data_stream)
    # Run the dl1 calibration (compute charge in photons)
    data_stream = dl1.calibrate_to_dl1(data_stream, time_integration_options, additional_mask=additional_mask, cleaning_threshold=5)

    data_stream = filter.filter_bigshower(data_stream, min_photon=5000)

    # Run the dl2 calibration (Hillas + classification + energy + direction)
    data_stream = dl2.calibrate_to_dl2(data_stream)

    ## Filter the events for display

    # data_stream = filter.filter_bigshower(data_stream, min_photon=0)

    with plt.style.context('ggplot'):
        display = EventViewer2(data_stream, n_samples=50, camera_config_file=camera_config_file, scale='lin')
        #display.next()
        display.draw()
        plt.show()
