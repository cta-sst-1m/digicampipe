from digicampipe.io import event_stream
from cts_core.camera import Camera
from digicampipe.utils import geometry
from digicamviewer.viewer import EventViewer
import matplotlib.pyplot as plt
from optparse import OptionParser


def main():
    parser = OptionParser()

    parser.add_option("-d", "--directory", dest="directory", help="directory to data files",
                      default='/home/alispach/data/BASELINE_TEST/')
    parser.add_option("-f", "--file",  dest="filename", help="filebasename e.g. CRAB_%03d.fits.fz",
                      default='BASELINE_TEST_0_000.%03d.fits.fz')
    parser.add_option('-s', "--file_start", dest='file_start', help='file number start', default=0)
    parser.add_option('-e', "--file_end", dest='file_end', help='file number end', default=23)
    parser.add_option('-c', "--camera_config", dest='camera_config_file', help='camera config file to load Camera()'
                      , default='/home/alispach/ctasoft/CTS/config/camera_config.cfg')

    (options, args) = parser.parse_args()
    read_raw_events(options, args)


def read_raw_events(options, args):
    directory = options.directory
    filename = directory + options.filename
    file_list = [filename % number for number in range(options.file_start, options.file_end + 1)]
    digicam_config_file = options.camera_config_file

    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    data_stream = event_stream.event_stream(file_list=file_list,
                                            camera_geometry=digicam_geometry,
                                            camera=digicam,
                                            expert_mode=True)

    with plt.style.context('ggplot'):
        display = EventViewer(data_stream, n_samples=50, camera_config_file=digicam_config_file, scale='lin')
        display.draw()
        plt.show()

    for data in data_stream:

        pass

if __name__ == '__main__':
    main()
