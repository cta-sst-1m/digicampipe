from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_dark import save_dark
from cts_core.camera import Camera
from digicampipe.utils import geometry
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in [0, 1, 2]]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_filename = 'dark.npz'

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.filter_event_types(data_stream, flags=[8])

    save_dark(data_stream, directory + dark_filename)

    data = np.load(directory + dark_filename)

    plt.figure()
    plt.hist(data['baseline'], bins='auto')
    plt.xlabel('dark baseline [LSB]')
    plt.ylabel('count')

    plt.figure()
    plt.hist(data['standard_deviation'], bins='auto')
    plt.xlabel('dark std [LSB]')
    plt.ylabel('count')
    plt.show()

    ## Filter the events for display
