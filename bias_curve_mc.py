from digicampipe.calib.camera import filter, r0, random_triggers
from digicampipe.io.save_bias_curve import save_bias_curve
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import geometry
from cts_core.camera import Camera
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    directory = '/home/alispach/data/digicamtoy/'
    filename = 'nsb_0.hdf5'
    file_list = [directory + filename]
    digicam_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'

    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)
    n_bins = 1024

    thresholds = np.arange(0, 400, 10)
    blinding = True
    trigger_filename = filename.strip('.hdf5') + '_bias_curve.npz'

    ####################

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera=digicam, camera_geometry=digicam_geometry, mc=True)

    data_stream = r0.fill_event_type(data_stream, flag=8)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = r0.fill_trigger_patch(data_stream)
    data_stream = r0.fill_trigger_input_7(data_stream)
    # data_stream = r0.fill_trigger_input_19(data_stream)
    data_stream = save_bias_curve(data_stream, thresholds=thresholds, blinding=blinding, output_filename=directory + trigger_filename)

    # for i, data in enumerate(data_stream):

      #  print(i)

    # trigger_mc = np.load(directory + 'trigger.npz')
    # trigger_mc_1 = np.load(directory + 'test.npz')
    # trigger_mc_2 = np.load(directory + 'trigger_non_uniform_nsb.npz')
    # trigger = np.load('/home/alispach/data/CRAB_01/' + 'trigger.npz')
    trigger_dark = np.load(directory + trigger_filename)

    fig = plt.figure()
    axis = fig.add_subplot(111)

    care_threshold = [100.,  120.,  140.,  160.,  180.,  200.]
    care_rate = [4.99999976e+06,   4.99999976e+06,   4.99999976e+06,   4.99843976e+06,  1.73769992e+05,   1.19999994e+02]
    care_err = [150755.66512837,  150755.66512837,  150755.66512837,  150732.14540993,  28104.5114053,   738.5489108 ]
    # axis.errorbar(trigger_mc['threshold'], trigger_mc['rate'] * 1E9, yerr=trigger_mc['rate_error'] * 1E9,
    #              label='DigicamToy $f_{nsb} = 1.2$ [GHz]')
    # axis.errorbar(trigger_mc_1['threshold'], trigger_mc_1['rate'] * 1E9, yerr=trigger_mc_1['rate_error'] * 1E9,
    #              label='DigicamToy New')
    # axis.errorbar(trigger['threshold'], trigger['rate'] * 1E9, yerr=trigger['rate_error'] * 1E9,
    #               label='Data', linestyle='--', color='k')
    # axis.errorbar(care_threshold, care_rate, yerr=care_err, label='CARE $f_{nsb} = 1.2$ [GHz]')
    # axis.errorbar(trigger_mc_2['threshold'], trigger_mc_2['rate'] * 1E9, yerr=trigger_mc_2['rate_error'] * 1E9,
    #              label='DigicamToy $f_{nsb} = 1.2 \pm 0.19$ [GHz]')
    axis.errorbar(trigger_dark['threshold'], trigger_dark['rate'] * 1E9, yerr=trigger_dark['rate_error'] * 1E9,
                  label='DigicamToy $f_{nsb}$ = 3 [MHz]')
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')
    # fig.savefig(directory + 'bias_curve.svg')

    plt.show()