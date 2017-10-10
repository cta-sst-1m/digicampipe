from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import event_stream
from digicampipe.io import save_external_triggers
from cts_core.camera import Camera
from digicampipe.utils import geometry
import astropy.units as u
from digicamviewer.viewer import EventViewer, EventViewer2
from digicampipe.utils import utils
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in range(3, 23)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    source_x = 0 * u.mm
    source_y = 0. * u.mm

    dark_baseline = np.load(directory + 'dark.npz')

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam, source_x=source_x, source_y=source_y)

    # Trigger configuration
    unwanted_patch = None

    """
    # Integration configuration
    time_integration_options = {'mask': None,
                                'mask_edges': None,
                                'peak': None,
                                'window_start': 3,
                                'window_width': 7,
                                'threshold_saturation': 3500,
                                'n_samples': 50,
                                'timing_width': 6,
                                'central_sample': 11}

    peak_position = utils.fake_timing_hist(time_integration_options['n_samples'], 
                                           time_integration_options['timing_width'],
                                           time_integration_options['central_sample'])
    
    temp = utils.generate_timing_mask(time_integration_options['window_start'], 
                                      time_integration_options['window_width'],
                                      peak_position)
    
    time_integration_options['peak'], time_integration_options['mask'], time_integration_options['mask_edges'] = temp

    """

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    # Fill the flags (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=1050)
    # Fill the baseline (to be replaced by Digicam)
    data_stream = filter.filter_event_types(data_stream, flags=[8])

    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)

    pixel_list = [0]

    next(data_stream)

    # save_external_triggers.save_external_triggers(data_stream, output_filename=directory + 'nsb.npz', pixel_list=pixel_list)

    data = np.load(directory + 'nsb.npz')

    plt.figure()
    plt.plot(data['time_stamp'])

    pixel = 0

    for key in data.keys():

        if key != 'time_stamp':

            plt.figure()
            plt.plot(data['time_stamp'] - data['time_stamp'][0], data[key][:, pixel],
                     label='pixel : {}'.format(pixel_list[pixel]))
            plt.xlabel('t [ns]')
            plt.ylabel(key)
            plt.legend()

            var = data[key][..., pixel]
            mask = ~np.isnan(var)

            plt.figure()
            plt.hist(var[mask], bins='auto')
            plt.xlabel(key)

    plt.show()


    ## Filter the events for display
