from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import geometry
from cts_core.camera import Camera
from digicamviewer.viewer import EventViewer
from os.path import expanduser
from optparse import OptionParser

if __name__ == '__main__':

    home_absolute_path = expanduser('~')
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="directory", help="directory to data files",
                      default='/sst1m/raw/2017/12/04/SST1M01/')
    parser.add_option('-f' , '--file', dest='filename', help='file basename e.g. CRAB_01_0_000.%03d.fits.fz',
                      default='SST1M01_20171204.%03d.fits.fz', type=str)
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
    pixel_not_wanted = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]

    ####################
    ##### ANALYSIS #####
    ####################

    # Define the event stream
    data_stream = event_stream(file_list=file_list, expert_mode=True, camera_geometry=digicam_geometry, camera=digicam)
    # Clean pixels

    data_stream = filter.set_pixels_to_zero(data_stream, unwanted_pixels=pixel_not_wanted)
    # Compute baseline with clocked triggered events (sliding average over n_bins)

    #with plt.style.context('ggplot'):
    display = EventViewer(data_stream, n_samples=50, camera_config_file=digicam_config_file, scale='lin')#, limits_colormap=[10, 500])
    display.draw()
        # pass