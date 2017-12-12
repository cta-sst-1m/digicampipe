from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import *
from digicampipe.utils import geometry
from cts_core.camera import Camera
from digicamviewer.viewer import EventViewer
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-c', "--camera_config", dest='camera_config_file', help='camera config file to load Camera()'
                      , default='/home/alispach/ctasoft/CTS/config/camera_config.cfg')
    (options, args) = parser.parse_args()
    # Input/Output files
    slow_control_file_list = ['./data/DigicamSlowControl_20171030_011.fits']
    drive_system_file_list = ['/mnt/sst1m_data/aux/2017/10/30/SST1M_01/DriveSystem_20171030_%03d.fits']
    file_list = ['./data/SST1M01_0_000.090.fits.fz']
    #slowcontrol_file_list=['/mnt/sst1m_data/aux/2017/10/30/SST1M_01/DigicamSlowControl_20171030_%03d.fits' % number for number in range(11 + 1)]
    #filename = options.directory + 'SST1M01_0_000.%03d.fits.fz'
    #file_list = [filename % number for number in range(options.file_start, options.file_end + 1)]
    digicam_config_file = options.camera_config_file
    # Camera and Geometry objects (mapping, pixel, patch + x,y coordinates pixels)
    digicam = Camera(_config_file=options.camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)
    # Config for NSB + baseline evaluation
    n_bins = 1000
    # Config for Hillas parameters analysis
    n_showers = 100000000
    reclean = True
    # Noisy patch that triggered
    unwanted_patch = None
    # Noisy pixels not taken into account in Hillas
    pixel_not_wanted = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]

    ####################
    ##### ANALYSIS #####
    ####################
    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry, camera=digicam)
    # Clean pixels
    data_stream = filter.set_pixels_to_zero(data_stream, unwanted_pixels=pixel_not_wanted)
    # Compute baseline with clocked triggered events (sliding average over n_bins)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    # Stop events that are not triggered by muons
    data_stream = filter.filter_event_types(data_stream, flags=[2])
    # Do not return events that have not the baseline computed (only first events)
    data_stream = filter.filter_missing_baseline(data_stream)
    #add slow data
    data_stream = add_slow_data(data_stream,
                                slow_control_file_list=slow_control_file_list,
                                drive_system_file_list=drive_system_file_list)


    ts_slow = []
    ts_data = []
    diff = []
    i = 0
    for event in data_stream:
        ts_slow.append(event.slowdata.slow_control.timestamp * 1e-3)
        ts_data.append(event.r0.tel[1].local_camera_clock * 1e-9)
        diff.append(ts_data[-1] - ts_slow[-1])
        i += 1
        if i == 100:
            i=0
            print(ts_slow[-1], ts_data[-1], diff[-1])

    from matplotlib import pyplot as plt

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ts_slow, ts_data)
    plt.subplot(2, 1, 2)
    plt.plot(diff)
    plt.show()
