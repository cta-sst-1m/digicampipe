import numpy as np
from digicampipe.io.event_stream import event_stream
import digicampipe.io.zfits as zfits
from digicampipe.utils import DigiCam
from digicamviewer.viewer import EventViewer
from os.path import expanduser
from optparse import OptionParser
import matplotlib.pyplot as plt
import os.path


if __name__ == '__main__':

    home_absolute_path = expanduser('~')
    parser = OptionParser()
    parser.add_option(
        "-p", "--path", dest="directory", help="directory to data files",
        default=os.path.realpath('digicampipe/tests/resources'))
    parser.add_option(
        '-f', '--file', dest='filename',
        help='file basename e.g. CRAB_01_0_000.%03d.fits.fz',
        default='/example_100_evts.%03d.fits.fz', type=str)
    parser.add_option(
        '-s', "--file_start", dest='file_start',
        help='file number start', default=0, type=int)
    parser.add_option(
        '-e', "--file_end", dest='file_end',
        help='file number end', default=0, type=int)

    (options, args) = parser.parse_args()

    # Input/Output files
    directory = options.directory
    filename = directory + options.filename
    file_list = [
        filename % number
        for number in range(options.file_start, options.file_end + 1)
    ]

    # Noisy pixels not taken into account
    # pixel_not_wanted = [
    #    1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    n_events = zfits.count_number_events(file_list=file_list)
    ####################
    ##### ANALYSIS #####
    ####################
    baseline = np.zeros((len(DigiCam.geometry.pix_id), n_events))

    # Define the event stream
    data_stream = event_stream(
        file_list=file_list,
        camera=DigiCam
    )

    for i, data in zip(range(n_events), data_stream):
        for tel_id in data.r0.tels_with_data:
            baseline[..., i] = data.r0.tel[tel_id].baseline

    plt.figure()
    plt.plot(baseline[0])
    plt.plot(baseline[1])
    plt.plot(baseline[2])
    plt.plot(baseline[1200])
    plt.show()

    display = EventViewer(
        data_stream,
        n_samples=50,
        camera_config_file=DigiCam.config_file,
        scale='lin',
        # limits_colormap=[10, 500])
    )
    display.draw()
