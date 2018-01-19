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


def extract_baseline(event_stream, calib_container):
    """
    Extract the baseline from events flagged as random trigger
        (trigger_flag = 1)
    :param event_stream: the event stream
    :param calib_container: the calibration container
    :return:
    """
    cc = calib_container  # use short form, for shorter lines

    for event in event_stream:

        # Check that the event is random trigger
        if event.trig.trigger_flag != 1:
            yield event

        for telid in event.r0.tels_with_data:
            # Get the adcs
            adcs = event.r0.tel[telid].adc_samples
            n_pixels, n_samples = adcs.shape
            samples = cc.samples_for_baseline
            # When the first event comes, add adcs.shape[-1] length
            # to the number of samples
            if cc.sample_to_consider == samples.shape[-1]:
                samples = np.append(
                    samples,
                    np.zeros_like(adcs),
                    axis=-1
                )

            # print(samples.shape)
            # Was the container filled up to n_samples_for_baseline?
            compute_full_baseline = True
            if cc.counter < samples.shape[-1] - 1:
                compute_full_baseline = False

            if cc.counter > 2 * n_samples:
                mask_noisy_pixels(cc, adcs)
            else:
                yield event

            # Insert new event
            if compute_full_baseline:
                if_case()
            else:
                else_case()
            yield event


def mask_noisy_pixels(cc, adcs):
    # Check the meaningfulness of previous event for baseline
    # calculation and set to nan noisy pixels
    # print(
    #     samples.shape,
    #     cc.counter,
    #     cc.counter + adcs.shape[-1]
    # )
    n_pixels, n_samples = adcs.shape
    samples = cc.samples_for_baseline

    slice_1 = slice(
        cc.counter - 1 * n_samples,
        cc.counter - 0 * n_samples)
    slice_2 = slice(
        cc.counter - 2 * n_samples,
        cc.counter - 1 * n_samples)
    prev_mean = samples[:, slice_1].mean(axis=-1)
    prevprev_mean = samples[:, slice_2].mean(axis=-1)
    present_mean = np.mean(adcs, axis=-1)

    some_unexplained_noise_limit = 50
    left_right_minimum = np.minimum(present_mean, prevprev_mean)

    noisy_pixels = (
        prev_mean - left_right_minimum >
        some_unexplained_noise_limit
    )
    samples[noisy_pixels, slice_1] = np.nan


def if_case(cc, adcs):
    n_pixels, n_samples = adcs.shape
    samples = cc.samples_for_baseline

    # shift all adcs by one event
    samples[:, :-n_samples] = (
        samples[:, n_samples:])
    # Add the new event
    samples[:, -n_samples:] = adcs
    # Compute the baseline and standard deviations
    cc.baseline_ready = True
    cc.baseline = np.nanmean(
        samples[:, :-n_samples],
        axis=-1)
    cc.std_dev = np.nanstd(
        samples[:, :-n_samples], axis=-1)


def else_case(cc, adcs):
    samples = cc.samples_for_baseline
    # Fill it in the proper place
    samples[
        :,
        cc.counter: cc.counter + adcs.shape[-1]
    ] = adcs
    # and increment the counter
    cc.counter += adcs.shape[-1]
    # Compute the baseline and standard deviations
    if cc.counter > 0.1*cc.sample_to_consider:
        cc.baseline_ready = True
        cc.baseline = np.nanmean(
            samples[:, :cc.counter-adcs.shape[-1]],
            axis=-1)
        cc.std_dev = np.nanstd(
            samples[:, :cc.counter-adcs.shape[-1]],
            axis=-1)
