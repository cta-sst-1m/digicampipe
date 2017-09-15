from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicamviewer.viewer import EventViewer
from digicampipe.utils import utils
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Data configuration
    directory = '/home/alispach/blackmonkey/calib_data/first_light/20170831/'
    filename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
    file_list = [filename % number for number in range(10, 165)]

    # Trigger configuration
    unwanted_patch = [391, 392, 403, 404, 405, 416, 417]

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

    peak_position = fake_timing_hist(time_integration_options['n_samples'], time_integration_options['timing_width'],
                                     time_integration_options['central_sample'])
    time_integration_options['peak'], time_integration_options['mask'], time_integration_options['mask_edges'] =\
        generate_timing_mask(time_integration_options['window_start'],
                             time_integration_options['window_width'],
                             peak_position)

    # Create the calibration container
    calib_data = initialise_calibration_data(n_samples_for_baseline = 10000)
    # Get the actual data stream
    data_stream = event_stream(file_list=file_list, expert_mode=True)
    # Filter events
    data_stream = filter.filter_patch(data_stream,unwanted_patch=unwanted_patch)
    # Deal with random trigger
    data_stream = random_triggers.extract_baseline(data_stream,calib_data)
    # Run the r1 calibration
    data_stream = r1.calibrate_to_r1(data_stream,calib_data,time_integration_options)

    n_events = 100000
    n_pixels = 1296

    time = np.zeros(n_events)

    for i, event, calib in zip(range(n_events), data_stream):
        camera_config_file = '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg'
        display = EventViewer(data_stream, camera_config_file=camera_config_file, scale='lin')
        display.draw()