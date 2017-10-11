from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicampipe.io import save_external_triggers
from cts_core.camera import Camera
from digicampipe.utils import geometry
import numpy as np


if __name__ == '__main__':
    # Data configuration

    directory_group = '/sst1m/raw/2017/09/28/CRAB_01/'
    filename = directory_group + 'CRAB_01_0_000.%03d.fits.fz'
    directory_local = '/home/alispach/'
    file_list = [filename % number for number in range(3, 4)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_baseline = np.load(directory_group + 'dark.npz')

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Trigger configuration
    unwanted_patch = None

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    # Fill the flags (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=2000)
    # Fill the baseline (to be replaced by Digicam)
    data_stream = filter.filter_event_types(data_stream, flags=[8])

    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)

    save_external_triggers.save_external_triggers(data_stream, output_filename=directory_local + 'nsb.npz')#, pixel_list=pixel_list)

    '''

    data = np.load(directory_local + 'nsb.npz')

    import matplotlib.pyplot as plt

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
    '''
