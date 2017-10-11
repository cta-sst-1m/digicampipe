from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_external_triggers import save_external_triggers
from cts_core.camera import Camera
from digicampipe.utils import geometry
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/' #
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in range(3, 23)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_baseline = np.load(directory + 'dark.npz')

    nsb_filename = 'nsb.npz'

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    pixel_list = np.arange(1296)

    # Trigger configuration
    unwanted_patch = None

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    # Fill the flags (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=1050)
    # Fill the baseline (to be replaced by Digicam)

    data_stream = filter.filter_missing_baseline(data_stream)

    data_stream = filter.filter_event_types(data_stream, flags=[8])

    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)

    data_stream = filter.filter_period(data_stream, period=10*u.second)

    save_external_triggers(data_stream, output_filename=directory + nsb_filename, pixel_list=pixel_list)

    data = np.load(directory + nsb_filename)

    plt.figure()
    plt.hist(data['baseline_dark'].ravel(), bins='auto', label='dark')
    plt.hist(data['baseline'].ravel(), bins='auto', label='nsb')
    plt.hist(data['baseline_shift'].ravel(), bins='auto', label='shift')
    plt.xlabel('baseline [LSB]')
    plt.ylabel('count')
    plt.legend()

    plt.figure()
    plt.hist(data['baseline_std'].ravel(), bins='auto', label='nsb')
    plt.xlabel('baseline std [LSB]')
    plt.ylabel('count')
    plt.legend()

    plt.figure()
    plt.hist(data['nsb_rate'][(data['nsb_rate'] > 0.) * (data['nsb_rate'] < 5)].ravel() * 1E3, bins='auto', label='all pixels')
    plt.xlabel('$f_{nsb}$ [MHz]')
    plt.ylabel('count')
    plt.legend()
    plt.show()

