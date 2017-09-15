import numpy as np
from digicampipe.io import event_stream, containers
from digicampipe.skimmer import skim_events, compute_discrimination_variable, compute_patch_coordinates


def filter_events(event_stream):

    patch_coordinates = compute_patch_coordinates()

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            # access the r0 container
            r0_container = event.r0.tel[telescope_id]
            # get informations on the event_type
            variable = compute_discrimination_variable(r0_container=r0_container, patch_coordinates=patch_coordinates)

            trigger_time, total_time_above_threshold, max_time_above_threshold, n_patches_above_threshold, sigma, nimp = variable

            if not nimp:
                # Set the event type
                event.trig.trigger_flag = 0
                yield event
            else:
                # Set the event type
                event.trig.trigger_flag = 1
                yield event


def analyse_random_trigger_events(event_stream, calib_stream):

    for event in event_stream:

        if event.trig.trigger_flag != 1 : continue

        # Get the adcs
        adcs = np.array(list(event.r0.tel[telid].adc_samples.values()))

        # Verfier ou on en est dans le counter

        # Checker si l'evenement precedent est pas chaud

        # Inserer l'evenement

        # Calculer baseline std gain drop








    return

def to_r1(event_stream):
    # Acceder au Gain (Cyril)
    # Acceder au XT, fonction de saturation (Cyril+Vic)

    # soustraction baseline (Vic)
    # integration (Vic)

    # calibration (/Gain/XT*sat)0

    return


def initialise_calibration_data(n_samples_for_baseline = 10000):
    '''
    Create a calibration data container to handle the data
    :param n_samples_for_baseline: Number of sample to evaluate the baseline
    :return:
    '''
    calib_container = containers.CalibrationDataContainer()
    calib_container.sample_to_consider = n_samples_for_baseline
    calib_container.samples_for_baseline = np.zeros((1296,n_samples_for_baseline),dtpye = int)
    calib_container.baseline = np.zeros((1296),dtpye = int)
    calib_container.std_dev = np.zeros((1296),dtpye = int)
    calib_container.counter = 0

    yield


if __name__ == '__main__':

    from digicamviewer.viewer import EventViewer

    camera_config_file = '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg'

    directory = '/home/alispach/blackmonkey/first_light/20170831/'
    filename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'

    file_list = [filename % number for number in range(30, 165)]

    # Some container to handle calibration data
    calib_data_holder = initialise_calibration_data(n_samples_for_baseline = 10000)
    # Get the raw data

    data_stream = event_stream(file_list=file_list, expert_mode=True)
    # Filter events
    data_stream = filter_events(data_stream)
    # Update the baseline
    data_stream,calib_data_holder = baseline_evaluation(data_stream,calib_data_holder)
    #  filtered_data = filter_events(event_stream)

#    data = skim_events(data_stream)
#    np.savez('temp.npz', **data)

    display = EventViewer(data_stream, camera_config_file=camera_config_file, scale='lin')
    display.draw()

    data = np.load('temp.npz')

    print(data['time_trigger'])

    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(data['time_trigger'], log=True)

    plt.figure()
    plt.hist(data['time_total'], log=True)

    plt.figure()
    plt.hist(data['time_max'], log=True)

    plt.figure()
    plt.hist(data['n_patches'], log=True)

    plt.figure()
    # plt.hist(data['shower_spread'], log=True)

    from scipy.stats import expon

    param = expon.fit(np.diff(data['time_trigger']), floc=0)
    plt.figure()
    hist = plt.hist(np.diff(data['time_trigger']), log=True)
    n_entries = np.sum(hist[0])
    bin_width = hist[1][1] - hist[1][0]
    pdf_fit = expon(loc=param[0], scale=param[1])
    plt.plot(hist[1], n_entries * bin_width * pdf_fit.pdf(hist[1]),
             label='$f_{trigger}$ = %0.2f [Hz]' % (1E9 / param[1]))
    plt.legend(loc='best')

    for key_1, val_1 in data.items():
        for key_2, val_2 in data.items():

            if key_1 in ['time_trigger', 'shower_spread'] or key_2 in ['time_trigger', 'shower_spread']:
              continue

            num = 10
            bins = [np.linspace(np.min(val_1), np.max(val_1), num=num), np.linspace(np.min(val_2), np.max(val_2), num=num)]
            plt.figure()
            plt.hist2d(val_1, val_2, bins=bins, norm=matplotlib.colors.LogNorm())
            plt.xlabel(key_1)
            plt.ylabel(key_2)

    plt.show()
