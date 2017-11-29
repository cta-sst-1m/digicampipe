from digicampipe.calib.camera import filter, r0
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import save_dark
from digicampipe.io.save_bias_curve import save_bias_curve
from cts_core.camera import Camera
from digicampipe.utils import geometry
import matplotlib.pyplot as plt
import numpy as np
from digicamviewer.viewer import EventViewer

if __name__ == '__main__':
    # Data configuration

    directory = '/home/alispach/data/CRAB_01/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in [3, ]]
    camera_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    dark_filename = 'dark.npz'

    trigger_filename = 'bias_curve_dark.npz'
    thresholds = np.arange(0, 400, 10)
    unwanted_patch = [306, 318, 330, 342, 200]
    pixel_not_wanted = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    unwanted_cluster = [200]
    blinding = True

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry, camera=digicam)
    data_stream = filter.set_pixels_to_zero(data_stream, unwanted_pixels=pixel_not_wanted)
    # data_stream = filter.set_patches_to_zero(data_stream, unwanted_patch=unwanted_patch)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.filter_event_types(data_stream, flags=[8])

    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = save_bias_curve(data_stream,
                                  thresholds=thresholds,
                                  blinding=blinding,
                                  output_filename=directory + trigger_filename,
                                  unwanted_cluster=unwanted_cluster)

    data_stream = save_dark(data_stream, directory + dark_filename)

    for i, data in enumerate(data_stream):

        print(i)
    """
    with plt.style.context('ggplot'):
        display = EventViewer(data_stream, n_samples=50, camera_config_file=camera_config_file,
                              scale='lin')  # , limits_colormap=[10, 500])
        display.draw()
        pass
    """

    data_dark = np.load(directory + dark_filename)
    data_rate = np.load(directory + trigger_filename)

    plt.figure()
    plt.hist(data_dark['baseline'], bins='auto')
    plt.xlabel('dark baseline [LSB]')
    plt.ylabel('count')

    plt.figure()
    plt.hist(data_dark['standard_deviation'], bins='auto')
    plt.xlabel('dark std [LSB]')
    plt.ylabel('count')


    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(data_rate['threshold'], data_rate['rate'] * 1E9, yerr=data_rate['rate_error'] * 1E9, label='Blinding : {}'.format(blinding))
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')
    plt.show()


    ## Filter the events for display
