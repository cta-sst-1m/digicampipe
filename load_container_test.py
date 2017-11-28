from digicampipe.calib.camera import r1, dl0, dl2, dl1
from digicampipe.io import containers
from digicamviewer.viewer import EventViewer
from digicampipe.utils import utils
import matplotlib.pyplot as plt


if __name__ == '__main__':
    camera_config_file = '/usr/src/cts/config/camera_config.cfg'
    # Integration configuration
    time_integration_options = {'mask': None,
                                'mask_edges': None,
                                'peak': None,
                                'window_start': 3,
                                'window_width': 7,
                                'threshold_saturation': 3500,
                                'n_samples': 50,
                                'timing_width': 6,
                                'central_sample': 11}

    peak_position = utils.fake_timing_hist(time_integration_options['n_samples'],
                                           time_integration_options['timing_width'],
                                           time_integration_options['central_sample'])
    time_integration_options['peak'], time_integration_options['mask'], time_integration_options['mask_edges'] = \
        utils.generate_timing_mask(time_integration_options['window_start'],
                                   time_integration_options['window_width'],
                                   peak_position)

    # Define the event stream
    data_stream = containers.load_from_pickle_gz('test.pickle')
    data_stream = r1.calibrate_to_r1(data_stream, None)
    data_stream = dl0.calibrate_to_dl0(data_stream)
    # Run the dl1 calibration (compute charge in photons)
    data_stream = dl1.calibrate_to_dl1(data_stream, time_integration_options)
#    data_stream = filter.filter_shower(data_stream, min_photon=1000)
    # Run the dl2 calibration (Hillas + classification + energy + direction)
    data_stream = dl2.calibrate_to_dl2(data_stream)

    with plt.style.context('ggplot'):
        display = EventViewer2(data_stream, n_samples=50, camera_config_file=camera_config_file, scale='lin')
        #display.next()
        display.draw()
        #plt.show()
