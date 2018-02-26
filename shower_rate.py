from digicampipe.calib.camera import filter, r1
from digicampipe.io.event_stream import event_stream
from digicampipe.visualization import EventViewer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon


if __name__ == '__main__':

    """
    directory = '/home/alispach/blackmonkey/calib_data/first_light/20170831/'
    filename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
    file_list = [filename % number for number in range(100, 165)]
    data_stream = event_stream(file_list=file_list, expert_mode=True)
    unwanted_patch = [380, 391, 392, 393, 403, 404, 405, 416, 417]

    data_stream = filter.filter_patch(data_stream, unwanted_patch=unwanted_patch, reversed=True)
    # data_stream = filter.filter_trigger_time(data_stream, time=10)

    n_events = 1000
    time = np.zeros(n_events)

    for i, event in zip(range(n_events), data_stream):

        for telescope_id in event.r0.tels_with_data:

            print(i)

            time[i] = event.r0.tel[telescope_id].local_camera_clock


    """
    directory = '/home/alispach/data/CRAB_01/'
    hillas_filename = directory + 'Merge_hillas_wo_corr.txt'

    hillas = np.genfromtxt(hillas_filename, names=['size', 'cen_x', 'cen_y', 'length', 'width', 'r', 'phi', 'psi', 'miss', 'alpha',
                                  'skewness', 'kurtosis', 'event_number', 'time_stamp'])

    time = hillas['time_stamp']
    time = np.sort(time)
    print(np.diff(time))

    plt.figure()
    plt.title('Cherenkov rate')
    hist = plt.hist(np.diff(time), bins=100, log=True, normed=False, align='mid')
    n_entries = np.sum(hist[0])
    bin_width = hist[1][1] - hist[1][0]
    param = expon.fit(np.diff(time), floc=0)
    pdf_fit = expon(loc=param[0], scale=param[1])
    plt.plot(hist[1], n_entries * bin_width * pdf_fit.pdf(hist[1]), label='$f_{trigger}$ = %0.2f [Hz]' % (1E9 / param[1]),
             linestyle='--')
    plt.xlabel('$\Delta t$ [ns]')
    plt.legend(loc='best')
    plt.show()

    """"
    display = EventViewer(data_stream, scale='lin')
    display.draw()
    plt.show()
    """
