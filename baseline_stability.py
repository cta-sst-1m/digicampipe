import numpy as np
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import geometry
from cts_core.camera import Camera
from digicamviewer.viewer import EventViewer
from os.path import expanduser
from optparse import OptionParser
import matplotlib.pyplot as plt
import os.path


if __name__ == '__main__':

    home_absolute_path = expanduser('~')
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="directory", help="directory to data files",
                      default=os.path.realpath('digicampipe/tests/resources'))
    parser.add_option('-f' , '--file', dest='filename', help='file basename e.g. CRAB_01_0_000.%03d.fits.fz',
                      default='example_100evts.000.fits.fz', type=str)
    # parser.add_option("-o", "--output", dest="output", help="output filename", default="output_crab.txt", type=str)
    # parser.add_option("-d", "--display", dest="display", action="store_true", help="Display rather than output data",
    #                   default=False)
    parser.add_option('-s', "--file_start", dest='file_start', help='file number start', default=0, type=int)
    parser.add_option('-e', "--file_end", dest='file_end', help='file number end', default=5, type=int)
    parser.add_option('-c', "--camera_config", dest='camera_config_file', help='camera config file to load Camera()'
                      , default=home_absolute_path + '/ctasoft/CTS/config/camera_config.cfg')

    (options, args) = parser.parse_args()

    # Input/Output files
    directory = options.directory
    filename = directory + options.filename
    file_list = [filename % number for number in range(options.file_start, options.file_end + 1)]
    digicam_config_file = options.camera_config_file

    # Camera and Geometry objects (mapping, pixel, patch + x,y coordinates pixels)
    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Noisy pixels not taken into account
    # pixel_not_wanted = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    n_events = 50000
    ####################
    ##### ANALYSIS #####
    ####################
    digicam_baseline = np.zeros((len(digicam_geometry.pix_id), n_events))

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry, camera=digicam)

    for i, data in zip(range(n_events), data_stream):

        for tel_id in data.r0.tels_with_data:

            digicam_baseline[..., i] = data.r0.tel[tel_id].digicam_baseline

    plt.figure()
    plt.plot(digicam_baseline[0])
    plt.plot(digicam_baseline[1])
    plt.plot(digicam_baseline[2])
    plt.plot(digicam_baseline[1200])
    plt.show()

    #with plt.style.context('ggplot'):
    display = EventViewer(data_stream, n_samples=50, camera_config_file=digicam_config_file, scale='lin')#, limits_colormap=[10, 500])
    display.draw()
        # pass