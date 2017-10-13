from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_external_triggers import save_external_triggers
from cts_core.camera import Camera
from digicampipe.utils import geometry
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from digicampipe.visualization import mpl


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

    # save_external_triggers(data_stream, output_filename=directory + nsb_filename, pixel_list=pixel_list)

    data = np.load(directory + nsb_filename)


    plt.figure()
    plt.hist(data['baseline_dark'].ravel(), bins='auto', label='dark')
    plt.hist(data['baseline'].ravel(), bins='auto', label='nsb')
    plt.hist(data['baseline_shift'].ravel(), bins='auto', label='shift')
    plt.xlabel('baseline [LSB]')
    plt.ylabel('count')
    plt.legend()

    baseline_change = np.diff(data['baseline'], axis=0)/np.diff(data['time_stamp'] * 1E-9)[:, np.newaxis]

    plt.figure()
    plt.hist(baseline_change.ravel(), bins='auto', label='pixel + time')
    plt.xlabel('dB/dt [LSB $\cdot$ s$^{-1}$ ]')
    plt.ylabel('count')
    plt.legend()

    baseline_max_min = np.max(data['baseline'], axis=0) - np.min(data['baseline'], axis=0)

    plt.figure()
    plt.hist(baseline_max_min, bins='auto', label='pixel')
    plt.xlabel('$B_{max} - B_{min}$ [LSB]')
    plt.ylabel('count')
    plt.legend()

    baseline_max_dark = np.max(data['baseline'], axis=0) - np.mean(data['baseline_dark'], axis=0)

    plt.figure()
    plt.hist(baseline_max_dark.ravel(), bins='auto', label='pixel')
    plt.xlabel('$B_{max} - B_{dark}$ [LSB]')
    plt.ylabel('count')
    plt.legend()

    plt.figure()
    plt.hist(data['baseline_std'].ravel(), bins='auto', label='pixel + time')
    plt.xlabel('baseline std [LSB]')
    plt.ylabel('count')
    plt.legend()

    mask = (baseline_max_min < 25) * (baseline_max_min > 0)
    mask = mask * (baseline_max_dark > 55) * (baseline_max_dark < 68)
    mask = mask[:, np.newaxis].T * np.abs(baseline_change) < 0.4
    mask = mask * (data['baseline_shift'][:-1] > 30) * (data['baseline_shift'][:-1] < 80)
    x = data['nsb_rate'][:-1]

    x = np.ma.array(x, mask=~mask)
    x = np.mean(x, axis=0)
    x = x[np.all(mask, axis=0)]

    #mask = np.all(mask, axis=0)
    #print(mask.shape)
    #x = x[:, mask]
    #print(x.shape)

    plt.figure()
    plt.hist(x, bins='auto', label='mean : {:.2f}, std : {:.2f}'.format(np.mean(x), np.std(x)))
    plt.xlabel('$f_{nsb}$ [GHz]')
    plt.ylabel('count')
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(data['nsb_rate'][(data['nsb_rate'] > 0.5) * (data['nsb_rate'] < 1.8)].ravel() * 1E3, bins='auto', label='all pixels')
    plt.xlabel('$f_{nsb}$ [MHz]')
    plt.ylabel('count')
    plt.legend()

    x = np.mean(data['nsb_rate'], axis=0)
    x = x[(x > 0.5) * (x < 1.8)]
    plt.figure()
    plt.hist(x, bins='auto', label='mean : {:.2f}, std : {:.2f}'.format(np.mean(x), np.std(x)))
    plt.xlabel('$f_{nsb}$ [GHz]')
    plt.ylabel('count')
    plt.legend()
    plt.show()



