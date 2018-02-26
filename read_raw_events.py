from digicampipe.io import event_stream
from cts_core.camera import Camera
from digicampipe.utils import geometry
from digicamviewer.viewer import EventViewer
import matplotlib.pyplot as plt
from optparse import OptionParser
from pkg_resources import resource_filename
from os import path

digicam_config_file = resource_filename(
    'digicampipe',
    path.join(
        'tests',
        'resources',
        'camera_config.cfg'
    )
)


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option(
        "-d", "--directory", dest="directory", help="directory to data files",
        default='/home/alispach/data/BASELINE_TEST/')
    parser.add_option(
        "-f", "--file",  dest="filename",
        help="filebasename e.g. CRAB_%03d.fits.fz",
        default='BASELINE_TEST_0_000.%03d.fits.fz')
    parser.add_option(
        '-s', "--file_start", dest='file_start',
        help='file number start', default=0)
    parser.add_option(
        '-e', "--file_end", dest='file_end',
        help='file number end', default=23)

    (options, args) = parser.parse_args()

    directory = options.directory
    filename = directory + options.filename
    urls = [filename % number for number in range(options.file_start,
                                                       options.file_end + 1)]

    data_stream = event_stream.event_stream(urls)
    with plt.style.context('ggplot'):
        display = EventViewer(
            data_stream,
            n_samples=50,
            camera_config_file=digicam_config_file,
            scale='lin'
        )
        display.draw()
        plt.show()

    for data in data_stream:

        pass
