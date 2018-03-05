import numpy as np


def fill_baseline_r0(event_stream, n_bins=10000):

    count_calib_events = 0

    for count, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            if count == 0:
                n_pixels = event.inst.num_pixels[telescope_id]
                n_samples = event.inst.num_samples[telescope_id]
                n_events = n_bins // n_samples
                baselines = np.zeros((n_pixels, n_events))
                baselines_std = np.zeros((n_pixels, n_events))
                baseline = np.zeros(n_pixels)
                std = np.zeros(n_pixels)

            r0_camera = event.r0.tel[telescope_id]

            if r0_camera.camera_event_type == 8:

                adc_samples = r0_camera.adc_samples
                new_mean = np.mean(adc_samples, axis=-1)
                new_std = np.std(adc_samples, axis=-1)

                baselines = np.roll(baselines, 1, axis=-1)
                baselines_std = np.roll(baselines_std, 1, axis=-1)

                baseline += new_mean - baselines[..., 0]
                std += new_std - baselines_std[..., 0]

                baselines[..., 0] = new_mean
                baselines_std[..., 0] = new_std

                count_calib_events += 1

            if count_calib_events >= n_events:

                r0_camera.baseline = baseline / n_events
                r0_camera.standard_deviation = std / n_events

            else:

                r0_camera.baseline = np.zeros(n_pixels) * np.nan
                r0_camera.standard_deviation = np.zeros(n_pixels) * np.nan

        yield event


def fill_baseline_r0_but_not_baseline(event_stream, n_bins=10000):

    count_calib_events = 0

    for count, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            if count == 0:
                n_pixels = event.inst.num_pixels[telescope_id]
                n_samples = event.inst.num_samples[telescope_id]
                n_events = n_bins // n_samples
                baselines = np.zeros((n_pixels, n_events))
                baselines_std = np.zeros((n_pixels, n_events))
                baseline = np.zeros(n_pixels)
                std = np.zeros(n_pixels)

            r0_camera = event.r0.tel[telescope_id]

            if r0_camera.camera_event_type == 8:

                adc_samples = r0_camera.adc_samples
                new_mean = np.mean(adc_samples, axis=-1)
                new_std = np.std(adc_samples, axis=-1)

                baselines = np.roll(baselines, 1, axis=-1)
                baselines_std = np.roll(baselines_std, 1, axis=-1)

                baseline += new_mean - baselines[..., 0]
                std += new_std - baselines_std[..., 0]

                baselines[..., 0] = new_mean
                baselines_std[..., 0] = new_std

                count_calib_events += 1

            if count_calib_events >= n_events:

                r0_camera.baseline = baseline / n_events
                r0_camera.standard_deviation = std / n_events

            else:

                r0_camera.baseline = np.zeros(n_pixels) * np.nan
                r0_camera.standard_deviation = np.zeros(n_pixels) * np.nan

            # do not even fill baseline with useful values
            r0_camera.baseline = np.zeros(n_pixels) * np.nan
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
            adcs = event.r0.tel[telid].adc_samples
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
