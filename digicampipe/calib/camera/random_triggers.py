import numpy as np
import digicampipe.io.containers as containers


def fill_baseline_r0(event_stream, n_bins=5000):

    n_pixels = 1296
    mean_temp = np.zeros(n_pixels)
    mean_new = np.zeros(n_pixels)
    std_temp = np.zeros(n_pixels)
    std_new = np.zeros(n_pixels)
    count_calib_events = 0

    for event_number, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]
            n_samples = r0_camera.num_samples
            n_events = n_bins // n_samples

            if r0_camera.flag == 0:

                adc_samples = np.array(list(r0_camera.adc_samples.values()))
                mean_temp += np.mean(adc_samples, axis=-1)
                std_temp += np.std(adc_samples, axis=-1)

                if (count_calib_events % n_events) == 0 and (count_calib_events > 0):

                    mean_new = mean_temp / n_events
                    std_new = std_temp / n_events
                    mean_temp = np.zeros(n_pixels)
                    std_temp = np.zeros(n_pixels)

                count_calib_events += 1

            r0_camera.baseline = mean_new
            r0_camera.standard_deviation = std_new

        yield event


def extract_baseline(event_stream, calib_container):
    """
    Extract the baseline from event flagged as random trigger (trigger_flag = 1)
    :param event_stream: the event stream
    :param calib_container: the calibration container
    :return:
    """

    pixel_list = list(range(1296))

    for event in event_stream:

        # Check that the event is random trigger
        if event.trig.trigger_flag != 1:
            yield event

        for telid in event.r0.tels_with_data:
            # Get the adcs
            adcs = np.array(list(event.r0.tel[telid].adc_samples.values()))
            # When the first event comes, add adcs.shape[-1] length to the number of samples
            if calib_container.sample_to_consider == calib_container.samples_for_baseline.shape[-1]:
                calib_container.samples_for_baseline = np.append(calib_container.samples_for_baseline,
                                                                 np.zeros((1296, adcs.shape[-1]), dtype=int),axis=-1)

            #print(calib_container.samples_for_baseline.shape)
            # Was the container filled up to n_samples_for_baseline?
            compute_full_baseline = True
            if calib_container.counter < calib_container.samples_for_baseline.shape[-1] - 1:
                compute_full_baseline = False

            # Check the meaningfulness of previous event for baseline calculation and set to nan noisy pixels
            if calib_container.counter - 2 * adcs.shape[-1] > 0 :
                #print(calib_container.samples_for_baseline.shape,calib_container.counter, calib_container.counter + adcs.shape[-1])
                prev_mean = np.mean(
                    calib_container.samples_for_baseline[:,
                    calib_container.counter - adcs.shape[-1]:calib_container.counter],
                    axis=-1)
                prevprev_mean = np.mean(calib_container.samples_for_baseline[:,
                                        calib_container.counter - 2 * adcs.shape[-1]:calib_container.counter] -
                                        adcs.shape[
                                            -1],
                                        axis=-1)
                present_mean = np.mean(adcs, axis=-1)
                calib_container.samples_for_baseline[prev_mean - np.minimum(present_mean, prevprev_mean) > 50][
                calib_container.counter - adcs.shape[-1]:calib_container.counter] = np.nan
            else:
                yield event

            # Insert new event
            if compute_full_baseline:
                # shift all adcs by one event
                calib_container.samples_for_baseline[:, :-adcs.shape[-1]] = calib_container.samples_for_baseline[:,
                                                                            adcs.shape[-1]:]
                # Add the new event
                calib_container.samples_for_baseline[:, -adcs.shape[-1]:] = adcs
                # Compute the baseline and standard deviations
                calib_container.baseline_ready = True
                calib_container.baseline = np.nanmean(calib_container.samples_for_baseline[:, :-adcs.shape[-1]],
                                                      axis=-1)
                calib_container.std_dev = np.nanstd(calib_container.samples_for_baseline[:, :-adcs.shape[-1]], axis=-1)
                yield event

            else:
                # Fill it in the proper place
                calib_container.samples_for_baseline[:,
                calib_container.counter: calib_container.counter + adcs.shape[-1]] = adcs
                # and increment the counter
                calib_container.counter += adcs.shape[-1]
                # Compute the baseline and standard deviations
                if calib_container.counter>0.1*calib_container.sample_to_consider :
                    calib_container.baseline_ready = True
                    calib_container.baseline = np.nanmean(calib_container.samples_for_baseline[:, :calib_container.counter-adcs.shape[-1]],
                                                          axis=-1)
                    calib_container.std_dev = np.nanstd(calib_container.samples_for_baseline[:, :calib_container.counter-adcs.shape[-1]], axis=-1)
                yield event


def initialise_calibration_data(n_samples_for_baseline = 10000):
    '''
    Create a calibration data container to handle the data
    :param n_samples_for_baseline: Number of sample to evaluate the baseline
    :return:
    '''
    calib_container = containers.CalibrationDataContainer()
    calib_container.sample_to_consider = n_samples_for_baseline
    calib_container.samples_for_baseline = np.zeros((1296,n_samples_for_baseline),dtype = int)
    calib_container.baseline = np.zeros((1296),dtype = int)
    calib_container.std_dev = np.zeros((1296),dtype = int)
    calib_container.counter = 0

    return calib_container