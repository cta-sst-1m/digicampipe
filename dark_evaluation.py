from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import event_stream
from digicamviewer.viewer import EventViewer, EventViewer2
from digicampipe.utils import utils
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Data configuration
    #directory = '/data/datasets/'
    # directory = '/home/alispach/blackmonkey/calib_data/first_light/20170831/'
    #directory = '/home/alispach/Downloads/'
    directory = '/calib_data/second_light/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in [0]]
    camera_config_file = '/data/software/CTS/config/camera_config.cfg'

    # Trigger configuration
    unwanted_patch = None #[391, 392, 403, 404, 405, 416, 417]

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
    data_stream = event_stream(file_list=file_list, expert_mode=True, geom_file=camera_config_file)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.fill_flag(data_stream , unwanted_patch=unwanted_patch)
    # Fill the baseline (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=18000)
    # Fill the baseline (to be replaced by Digicam)
    data_stream = random_triggers.dump_baseline(data_stream, '/calib_data/second_light/dark.npz', n_bins=18000)

    ## Filter the events for display
