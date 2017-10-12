from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io import save_bias_curve
from cts_core.camera import Camera
from digicampipe.utils import geometry
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/'  #
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in range(3, 23)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Trigger configuration
    unwanted_patch = [306, 318, 330, 342]

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    data_stream = filter.filter_event_types(data_stream, flags=[8])

    save_bias_curve.save_bias_curve(data_stream, output_filename=directory + 'trigger_rate.npz', camera=digicam, unwanted_patch=unwanted_patch)

    data = np.load(directory + 'trigger_rate.npz')

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(data['threshold'], data['rate'] * 1E9, yerr=data['rate_error'])
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')
    plt.show()