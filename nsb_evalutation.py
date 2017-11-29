from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_external_triggers import save_external_triggers
from cts_core.camera import Camera
from digicampipe.utils import geometry, utils
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from digicampipe.visualization import mpl
import pandas as pd


if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/' #
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in range(3, 23)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_baseline = np.load(directory + 'dark.npz')

    # nsb_filename = 'nsb.npz'
    nsb_filename = 'nsb_with_std.npz'

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    pixel_list = np.arange(1296)

    # Trigger configuration
    unwanted_patch = None

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry, camera=digicam)
    # Fill the flags (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=1050)
    # Fill the baseline (to be replaced by Digicam)

    data_stream = filter.filter_missing_baseline(data_stream)

    data_stream = filter.filter_event_types(data_stream, flags=[8])

    # data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)
    data_stream = r1.calibrate_to_r1(data_stream)

    data_stream = filter.filter_period(data_stream, period=10*u.second)

    # save_external_triggers(data_stream, output_filename=directory + nsb_filename, pixel_list=pixel_list)

    data = np.load(directory + nsb_filename)

    print(data.__dict__)

    '''
    plt.figure()
    plt.hist(data['baseline_dark'].ravel(), bins='auto', label='dark')
    plt.hist(data['baseline'].ravel(), bins='auto', label='nsb')
    plt.hist(data['baseline_shift'].ravel(), bins='auto', label='shift')
    plt.xlabel('baseline [LSB]')
    plt.ylabel('count')
    plt.legend()
    '''
    '''

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

    '''
    sectors = [1, 3]
    pixel_sector_1 = [pixel.ID for pixel in digicam.Pixels if pixel.sector == 1]
    pixel_sector_2 = [pixel.ID for pixel in digicam.Pixels if pixel.sector == 2]
    pixel_sector_3 = [pixel.ID for pixel in digicam.Pixels if pixel.sector == 3]

    x_1 = data['nsb_rate'].T[pixel_sector_1]
    x_2 = data['nsb_rate'].T[pixel_sector_2]
    x_3 = data['nsb_rate'].T[pixel_sector_3]

    # mask = np.where(x_1 > 0 * x_1 < 2)[0]
    # nsb_time = x_1[mask]
    # print(nsb_time)
    nsb_time_1 = np.mean(x_1, axis=-1)
    print(nsb_time_1)
    mask_1 = (nsb_time_1 < 2) * (nsb_time_1 > 0.75)
    nsb_time_1 = nsb_time_1[mask_1]
    nsb_time_2 = np.mean(x_2, axis=-1)
    mask_2 = (nsb_time_2 < 2) * (nsb_time_2 > 0.75)
    nsb_time_2 = nsb_time_2[mask_2]
    nsb_time_3 = np.mean(x_3, axis=-1)
    mask_3 = (nsb_time_3 < 2) * (nsb_time_3 > 0.75)
    nsb_time_3 = nsb_time_3[mask_3]
    time = data['time_stamp']

    plt.figure()
    plt.plot(time[mask_1] * 1E-9, nsb_time_1, label='sector 1')
    plt.plot(time[mask_1] * 1E-9, utils.moving_average(nsb_time_1, n=100) , label='sector 1')
    # plt.plot(time[mask_2] * 1E-9, nsb_time_2, label='sector 2')
    # plt.plot(time[mask_2] * 1E-9, utils.moving_average(nsb_time_2, n=100), label='sector 2')
    plt.plot(time[mask_3] * 1E-9, nsb_time_3, label='sector 3')
    plt.plot(time[mask_3] * 1E-9, utils.moving_average(nsb_time_3, n=100), label='sector 3')
    plt.xlabel('time [s]')
    plt.ylabel('$f_{nsb}$ [GHz]')
    plt.legend()
    plt.show()

    x_1 = np.mean(x_1, axis=-1)
    x_2 = np.mean(x_2, axis=-1)
    x_3 = np.mean(x_3, axis=-1)

    indices_to_keep = np.where(x_1 > 0)
    x_1 = x_1[indices_to_keep[0]]
    indices_to_keep = np.where(x_2 > 0)
    x_2 = x_2[indices_to_keep[0]]
    indices_to_keep = np.where(x_3 > 0)
    x_3 = x_3[indices_to_keep[0]]

    temp = np.where(((data['baseline_shift'].T <= 0) + (data['baseline_shift'].T >= 80)) > 0)
    print(temp[0].shape, temp[1].shape)

    plt.figure()
    plt.hist(temp[0], bins=np.arange(0, 1296, 1))

    n_bins = np.arange(0.5, 2 + 0.05, 0.05)

    plt.figure()
    plt.hist(x_1.ravel(), bins=n_bins, label='sector : {} \n mean : {:0.2f} \n std : {:0.2f}'.format(1, np.mean(x_1), np.std(x_1)), alpha=0.5)
    # plt.hist(x_2.ravel(), bins='auto', label='sector : {} \n mean : {:0.2f} \n std : {:0.2f}'.format(2, np.mean(x_2), np.std(x_2)))
    plt.hist(x_3.ravel(), bins=n_bins, label='sector : {} \n mean : {:0.2f} \n std : {:0.2f}'.format(3, np.mean(x_3), np.std(x_3)), alpha=0.5)
    plt.legend()
    plt.xlabel('$f_{nsb} [GHz]$')
    plt.show()

    mask = (baseline_max_min < 25) * (baseline_max_min > 0)
    mask = mask * (baseline_max_dark > 55) * (baseline_max_dark < 60)
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



