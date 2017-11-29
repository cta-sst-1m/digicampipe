from digicampipe.calib.camera import filter, random_triggers, r1, dl0, dl2, dl1
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import utils, geometry
from cts_core.camera import Camera
import astropy.units as u
from digicampipe.io.containers import save_to_pickle_gz

def main():
    # Data configuration
    directory = '/mnt/calib_data/first_light/20170831/'
    filename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
#    file_list = [filename % number for number in range(110,120)]
    file_list = [filename % 110]
    camera_config_file = '/usr/src/cts/config/camera_config.cfg'
    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam, source_x=0 * u.mm, source_y=0. * u.mm)

    # Trigger configuration
    unwanted_patch = [391, 392, 403, 404, 405, 416, 417]

    # Integration configuration
    time_integration_options = {'mask':None,
                                'mask_edges':None,
                                'peak':None,
                                'window_start':3,
                                'window_width':7,
                                'threshold_saturation':3500,
                                'n_samples':50,
                                'timing_width':6,
                                'central_sample':11}

    peak_position = utils.fake_timing_hist(time_integration_options['n_samples'], time_integration_options['timing_width'],
                                     time_integration_options['central_sample'])
    time_integration_options['peak'], time_integration_options['mask'], time_integration_options['mask_edges'] = \
        utils.generate_timing_mask(time_integration_options['window_start'],
                             time_integration_options['window_width'],
                             peak_position)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, camera_geometry=digicam_geometry, expert_mode=True)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.fill_flag(data_stream , unwanted_patch=unwanted_patch)
    # Fill the baseline (to be replaced by Digicam)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=3000)
#    data_stream = filter.filter_event_types(data_stream, flags=[1])
    data_stream = filter.filter_missing_baseline(data_stream)

    save_to_pickle_gz(data_stream, 'test.pickle', overwrite=True, max_events=100)

if __name__ == '__main__':
    main()
