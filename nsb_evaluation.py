from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from cts_core.camera import Camera
from digicampipe.utils import geometry
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in range(3, 23)]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_baseline = np.load(directory + 'dark.npz')

    nsb_filename = 'nsb.npz'

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    pixel_list = np.arange(1296)

    data_stream = event_stream(
        file_list=file_list,
        expert_mode=True,
        camera_geometry=digicam_geometry
    )
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=1050)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)
    data_stream = filter.filter_period(data_stream, period=10*u.second)

    data = np.load(directory + nsb_filename)

    plt.figure()
    plt.hist(data['baseline_dark'].ravel(), bins='auto', label='dark')
    plt.hist(data['baseline'].ravel(), bins='auto', label='nsb')
    plt.hist(data['baseline_shift'].ravel(), bins='auto', label='shift')
    plt.xlabel('baseline [LSB]')
    plt.ylabel('count')
    plt.legend()

    sectors = [1, 3]
    pixel_not_in_intersil = [pixel.ID for pixel in digicam.Pixels if pixel.sector in sectors]

    print(len(pixel_not_in_intersil))

    x = data['nsb_rate'].T[pixel_not_in_intersil]

    x = np.mean(x, axis=-1)

    indices_to_keep = np.where(x > 0)
    x = x[indices_to_keep[0]]

    temp = np.where(
        (
            (data['baseline_shift'].T <= 0) +
            (data['baseline_shift'].T >= 80)
        ) > 0
    )
    print(temp[0].shape, temp[1].shape)

    plt.figure()
    plt.hist(temp[0], bins=np.arange(0, 1296, 1))

    plt.figure()
    plt.hist(x.ravel(), bins='auto', label='sector : {} \n mean : {:0.2f} \n std : {:0.2f}'.format(sectors, np.mean(x), np.std(x)))
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
    plt.hist(
        data['nsb_rate'][
            (data['nsb_rate'] > 0.5) * (data['nsb_rate'] < 1.8)
        ].ravel() * 1E3,
        bins='auto',
        label='all pixels'
    )
    plt.xlabel('$f_{nsb}$ [MHz]')
    plt.ylabel('count')
    plt.legend()

    x = np.mean(data['nsb_rate'], axis=0)
    x = x[(x > 0.5) * (x < 1.8)]
    plt.figure()
    plt.hist(
        x,
        bins='auto',
        label='mean : {:.2f}, std : {:.2f}'.format(
            np.mean(x),
            np.std(x)
        )
    )
    plt.xlabel('$f_{nsb}$ [GHz]')
    plt.ylabel('count')
    plt.legend()
    plt.show()
