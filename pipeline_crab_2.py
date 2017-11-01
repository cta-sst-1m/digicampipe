from digicampipe.io import event_stream
from digicampipe.calib.camera import filter, random_triggers, r0, r1
from digicampipe.io.save_adc import save_dark
from digicampipe.io.save_external_triggers import save_external_triggers
from cts_core.camera import Camera
from digicampipe.utils import geometry
from digicamviewer.viewer import EventViewer
import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
import astropy.units as u


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("-d", "--directory", dest="directory", help="directory to data files",
                      default='/home/alispach/data/CRAB_02/')
    parser.add_option("-f", "--file",  dest="filename", help="filebasename e.g. CRAB_%03d.fits.fz",
                      default='SST1M01_0_000.%03d.fits.fz')
    parser.add_option('-s', "--file_start", dest='file_start', help='file number start', default=0, type=int)
    parser.add_option('-e', "--file_end", dest='file_end', help='file number end', default=100, type=int)
    parser.add_option('-c', "--camera_config", dest='camera_config_file', help='camera config file to load Camera()'
                      , default='/home/alispach/ctasoft/CTS/config/camera_config.cfg')

    (options, args) = parser.parse_args()

    directory = options.directory
    filename = directory + options.filename
    file_list = [filename % number for number in range(options.file_start, options.file_end + 1)]
    file_list_dark = [filename % number for number in range(1, 4 + 1)]
    print(file_list_dark)
    digicam_config_file = options.camera_config_file
    dark_filename = 'dark.npz'
    nsb_filename = 'nsb.npz'
    n_bins = 2000

    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    unwanted_pixels = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896, 1196, 1172, 1173, 1146, 1117, 1118, 1085, 1197, 1198, 1174, 1175, 1148, 1149, 1119, 1147, 1121, 1120, 1150, 1177, 1176, 1200, 1199, 1220, 1219, 1239, 1221, 1222, 1201, 1202, 1178, 1151, 1152, 1179, 1240, 1241, 1203, 1204, 1180, 1181, 1206, 1205, 1226, 1225, 1243, 1242, 1257, 1256, 1223, 1224]

    data_stream = event_stream.event_stream(file_list=file_list_dark,
                                            camera_geometry=digicam_geometry,
                                            camera=digicam,
                                            expert_mode=True)

    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = filter.set_pixels_to_zero(data_stream, unwanted_pixels=unwanted_pixels)
    data_stream = save_dark(data_stream, directory + dark_filename)

    i = 0
    for data in data_stream:

         print(i)
         i += 1

    data_dark = np.load(directory + dark_filename)

    print(data_dark.__dict__)


    """
    data_stream = event_stream.event_stream(file_list=file_list,
                                            camera_geometry=digicam_geometry,
                                            camera=digicam,
                                            expert_mode=True)

    data_stream = filter.set_pixels_to_zero(data_stream, unwanted_pixels=unwanted_pixels)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = r1.calibrate_to_r1(data_stream, data_dark)
    data_stream = filter.filter_period(data_stream, period=10*u.second)
    data_stream = save_external_triggers(data_stream, output_filename=directory + nsb_filename, yielding=False)

    # for data in data_stream:

    #    pass

    # with plt.style.context('ggplot'):
    #    display = EventViewer(data_stream, n_samples=50, camera_config_file=digicam_config_file, scale='lin')
    #    display.draw()

    """
    data_nsb = np.load(directory + nsb_filename)

    plt.figure()
    plt.hist(data_dark['baseline'], bins='auto')
    plt.xlabel('dark baseline [LSB]')
    plt.ylabel('count')

    plt.figure()
    plt.hist(data_dark['standard_deviation'], bins='auto')
    plt.xlabel('dark std [LSB]')
    plt.ylabel('count')

    plt.figure()
    plt.hist(data_nsb['baseline_dark'].ravel(), bins='auto', label='dark', alpha=0.3)
    plt.hist(data_nsb['baseline'].ravel(), bins='auto', label='nsb', alpha=0.3)
    plt.hist(data_nsb['baseline_shift'].ravel(), bins='auto', label='shift', alpha=0.3)
    plt.xlabel('baseline [LSB]')
    plt.ylabel('count')
    plt.legend()

    x = data_nsb['nsb_rate'].ravel()
    mask = (x > 0) *( x < 3)
    x = x[mask]

    plt.figure()
    plt.hist(x, bins=50, label='nsb', alpha=0.3)
    plt.xlabel('$f_{nsb}$ [GHz]')
    plt.ylabel('count')
    plt.legend()

    plt.show()