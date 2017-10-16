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
    unwanted_patch = None # [306, 318, 330, 342, 200]
    blinding = True
    by_cluster = True
    thresholds = np.arange(0, 500, 5)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    # data_stream = filter.set_patches_to_zero(data_stream, unwanted_patch=unwanted_patch)
    data_stream = save_bias_curve(data_stream, thresholds=thresholds,
                                  output_filename=directory + trigger_filename,
                                  camera=digicam,
                                  blinding=blinding,
                                  by_cluster=by_cluster)

    for i, data in enumerate(data_stream):

        print(i)

    data = np.load(directory + trigger_filename)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(data['threshold'], data['rate'] * 1E9, yerr=data['rate_error'] * 1E9, label='Blinding : {}'.format(blinding))
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')

    if by_cluster:

        fig = plt.figure()
        axis = fig.add_subplot(111)

        n_clusters = data['cluster_rate'].shape[0]
        for i in range(n_clusters):

            axis.plot(data['threshold'], data['cluster_rate'][i] * 1E9, label='Cluster : {}'.format(i))

        axis.set_ylabel('rate [Hz]')
        axis.set_xlabel('threshold [LSB]')
        axis.set_yscale('log')
        axis.legend(loc='best')

    plt.show()