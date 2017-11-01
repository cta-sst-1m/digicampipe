from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import save_dark
from digicampipe.io.save_bias_curve import save_bias_curve
from cts_core.camera import Camera
from digicampipe.utils import geometry
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Data configuration

    directory = '/Users/nagai/ctasoft/data/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in [1]]
    camera_config_file = '/Users/nagai/ctasoft/CTS/config/camera_config.cfg'
    dark_filename = 'dark.npz'

    thresholds = np.arange(0, 400, 10)
    unwanted_patch = [306, 318, 330, 342, 200]
    unwanted_cluster = [200]
    blinding = True

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, camera=digicam, expert_mode=True, camera_geometry=digicam_geometry)
    data_stream = filter.set_patches_to_zero(data_stream, unwanted_patch=unwanted_patch)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = save_dark(data_stream, directory + dark_filename)

    # for i, data in enumerate(data_stream):

    #    print(i)

    data_dark = np.load(directory + dark_filename)

    plt.figure()
    plt.hist(data_dark['baseline'], bins='auto')
    plt.xlabel('dark baseline [LSB]')
    plt.ylabel('count')

    plt.figure()
    plt.hist(data_dark['standard_deviation'], bins='auto')
    plt.xlabel('dark std [LSB]')
    plt.ylabel('count')

    plt.show()


    ## Filter the events for display
