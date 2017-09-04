import numpy as np
from event_stream import event_stream


def filter_events(event_stream):

    data_to_return = []

    for event in event_stream:

        n_patches = 100

        for telescope_id in event.r0.tels_with_data:

            output_trigger_patch7 = np.array(list(event.r0.tel[telescope_id].trigger_output_patch7.values()))

            if np.sum(output_trigger_patch7) > n_patches:
                data = np.array(list(event.r0.tel[telescope_id].adc_samples.values()))
                data_to_return.append(data_to_return)
                print(data)

    return np.array(data_to_return)


if __name__ == '__main__':

    from stats import stats_events

    directory = '/home/alispach/Downloads/'
    filename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
    file_list = [filename %number for number in range(102, 103)]
    data_stream = event_stream(file_list=file_list, expert_mode=True)
    #  filtered_data = filter_events(event_stream)

    stats = stats_events(data_stream)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(stats[0], log=True)

    plt.figure()
    plt.hist(stats[1], log=True)

    from scipy.stats import expon

    param = expon.fit(np.diff(stats[2]), floc=0)
    plt.figure()
    hist = plt.hist(np.diff(stats[2]), log=True)
    n_entries = np.sum(hist[0])
    bin_width = hist[1][1] - hist[1][0]
    pdf_fit = expon(loc=param[0], scale=param[1])
    plt.plot(hist[1], n_entries * bin_width * pdf_fit.pdf(hist[1]),
             label='$f_{trigger}$ = %0.2f [Hz]' % (1E9 / param[1]))
    plt.legend(loc='best')
    plt.show()