from digicampipe.calib.camera import filter, r0, random_triggers
from digicampipe.io.save_bias_curve import save_bias_curve
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import geometry
from cts_core.camera import Camera
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    directory = '/home/alispach/data/digicam_commissioning/trigger/mc/'
    filename = directory + 'nsb_full_camera_105.hdf5'
    file_list = [filename]
    digicam_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'

    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)
    n_bins = 1024

    thresholds = np.arange(0, 400, 10)
    blinding = True
    trigger_filename = 'trigger_non_uniform_nsb.npz'

    ####################

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera=digicam, camera_geometry=digicam_geometry)

    data_stream = r0.fill_event_type(data_stream, flag=8)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = r0.fill_trigger_patch(data_stream)
    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = r0.fill_trigger_input_19(data_stream)
    data_stream = save_bias_curve(data_stream, thresholds=thresholds, blinding=blinding, output_filename=directory + trigger_filename)

    for i, data in enumerate(data_stream):

        print(i)

    trigger_mc = np.load(directory + 'trigger.npz')
    trigger_mc_2 = np.load(directory + 'trigger_non_uniform_nsb.npz')
    trigger = np.load('/home/alispach/data/CRAB_01/' + 'trigger.npz')

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(trigger_mc['threshold'], trigger_mc['rate'] * 1E9, yerr=trigger_mc['rate_error'] * 1E9,
                  label='DigicamToy')
    axis.errorbar(trigger['threshold'], trigger['rate'] * 1E9, yerr=trigger['rate_error'] * 1E9,
                  label='Data')
    axis.errorbar(trigger_mc_2['threshold'], trigger_mc_2['rate'] * 1E9, yerr=trigger_mc_2['rate_error'] * 1E9,
                  label='DigicamToy NSB 1.2 +- 0.19 [GHz]')
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')
    # fig.savefig(directory + 'bias_curve.svg')
    plt.show()