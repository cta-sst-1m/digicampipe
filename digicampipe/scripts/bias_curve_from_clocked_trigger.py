from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_bias_curve import save_bias_curve
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
    trigger_filename = 'trigger_rate_no_blinding.npz'

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Trigger configuration
    unwanted_patch = None  # [306, 318, 330, 342, 200]
    blinding = True
    by_cluster = False
    thresholds = np.arange(0, 500, 5)

    rate_care = [[2.50000015e+08, 2.50000015e+08, 2.50000015e+08, 2.50000015e+08, 1.50675009e+08, 3.55000021e+06,
                  2.50000015e+04],
                 [2.50000015e+08, 2.50000015e+08, 2.50000015e+08, 2.50000015e+08, 2.49125015e+08, 5.25000031e+07,
                  1.55000009e+06],
                 [2.50000015e+08, 2.50000015e+08, 2.50000015e+08, 2.50000015e+08, 2.32075014e+08, 6.01250036e+07,
                  4.52500027e+07, 4.50000027e+07, 4.50000027e+07, 4.50000027e+07, 4.50000027e+07, 4.50000027e+07]]

    rate_error_care = [[1.11803406e+08, 1.11803406e+08, 1.11803406e+08, 1.11803406e+08, 8.67971825e+07, 1.33229134e+07,
                        1.11803406e+06],
                       [1.00000006e+08, 1.00000006e+08, 1.00000006e+08, 1.00000006e+08, 9.98248525e+07, 4.58257597e+07,
                        7.87400834e+06],
                       [1.44337576e+08, 1.44337576e+08, 1.44337576e+08, 1.44337576e+08, 1.39066839e+08, 7.07843010e+07,
                        6.14071151e+07, 6.12372472e+07, 6.12372472e+07, 6.12372472e+07, 6.12372472e+07, 6.12372472e+07]]
    threshold_care = [[10., 25., 40., 55., 70., 85., 100.],
                      [10., 20., 30., 40., 50., 60., 70.],
                      [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.]]
    nsb_care = [1.5, 1.0, 0.9]

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    # data_stream = filter.set_patches_to_zero(data_stream, unwanted_patch=unwanted_patch)
    # data_stream = save_bias_curve(data_stream, thresholds=thresholds,
    #                              output_filename=directory + trigger_filename,
    #                              camera=digicam,
    #                              blinding=blinding,
    #                              by_cluster=by_cluster)

    # for i, data in enumerate(data_stream):

    #    print(i)

    data = np.load(directory + trigger_filename)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(data['threshold'], data['rate'] * 1E9, yerr=data['rate_error'] * 1E9,
                  label='Blinding : {}'.format(blinding))
    axis.errorbar(threshold_care[0], rate_care[0], yerr=rate_error_care[0], label='Care : {} [GHz]'.format(nsb_care[0]))
    axis.errorbar(threshold_care[1], rate_care[1], yerr=rate_error_care[1], label='Care : {} [GHz]'.format(nsb_care[1]))
    axis.errorbar(threshold_care[2], rate_care[2], yerr=rate_error_care[2], label='Care : {} [GHz]'.format(nsb_care[2]))
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')

    if by_cluster:

        fig = plt.figure()
        axis = fig.add_subplot(111)

        n_clusters = data['cluster_rate'].shape[0]
        for i in [306, 318, 330, 342, 200, 100]:
            axis.errorbar(data['threshold'], data['cluster_rate'][i] * 1E9, yerr=data['cluster_rate_error'][i] * 1E9,
                          label='{} cluster : {}'.format('noisy' if i != 100 else 'regular', i))

        axis.set_ylabel('rate [Hz]')
        axis.set_xlabel('threshold [LSB]')
        axis.set_yscale('log')
        axis.legend(loc='best')

    plt.show()
