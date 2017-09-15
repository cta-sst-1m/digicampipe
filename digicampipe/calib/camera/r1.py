import numpy as np


def calibrate_to_r1(event_stream, calib_container):
    cleaning_threshold = 3.

    pixel_list = list(range(1296))

    for event in event_stream:
        # Check that the event is physics trigger
        if event.trig.trigger_flag != 0:
            yield event
            continue
        # Check that there were enough random trigger to compute the baseline
        if not calib_container.baseline_ready :
            yield event
            continue

        for telescope_id in event.r0.tels_with_data:
            # Get the R0 and R1 containers
            r0_camera = event.r0.tel[telescope_id]
            r1_camera = event.r1.tel[telescope_id]
            # Get the ADCs
            adc_samples = np.array(list(r0_camera.adc_samples.values()))
            # Get the mean and standard deviation
            r1_camera.pedestal_mean = calib_container.baseline
            r1_camera.pedestal_std = calib_container.std_dev
            # Subtract baseline to the data
            adc_samples = adc_samples - r1_camera.pedestal_mean
            # Compute the gain drop and NSB
            if calib_container.dark_baseline is None :
                # compute NSB and Gain drop from STD
                r1_camera.gain_drop =
                r1_camera.nsb  = np.ones(adc_samples.shape[0]) * 1.e9
            else:
                # compute NSB and Gain drop from baseline shift
                r1_camera.gain_drop = np.ones(adc_samples.shape[0]) * 1.
                r1_camera.nsb  = np.ones(adc_samples.shape[0]) * 1.e9

            gain_init = np.ones(adc_samples.shape[0]) * 23. # TODO, replace gain of 23 by calib array of gain
            gain = gain_init * r1_camera.gain_drop

            # mask pixels which goes above N sigma
            mask_for_cleaning = adcs_samples > cleaning_threshold  * r1_camera.pedestal_std
            mask_for_cleaning = np.any(mask_for_cleaning,axis=-1)

            # Integrate the data
            adc_samples = integrate(adc_samples, time_integration_options['window_width'])

            # Compute the charge
            charge = extract_charge(adc_samples, time_integration_options['mask'],
                                    time_integration_options['mask_edges'],
                                    time_integration_options['peak'],
                                    time_integration_options['window_start'],
                                    time_integration_options['threshold_saturation'])

            r1_camera.pe_samples = dict(zip(pixel_list, charge))


            yield event


# Define the integration function
def integrate(data, window_width):
    """
    Simple integration function over N samples

    :param data:
    :param options:
    :return:
    """
    if window_width == 1 : return data
    h = ndimage.convolve1d(data,np.ones(window_width, dtype=int),axis=-1,mode='constant',cval=-1.e8)
    return h[...,int(np.floor((window_width-1)/2)):-int(np.floor(window_width/2))]


def extract_charge(adc_samples, mask, mask_edges, peak, window_start, threshold_saturation):
    """
    Extract the charge.
       - check which pixels are saturated
       - get the local maximum within the timing mask and check if it is not at the edge of the mask
       - move window_start from the maximum
    :param data:
    :param timing_mask:
    :param timing_mask_edge:
    :param peak_position:
    :param options:
    :param integration_type:
    :return:
    """
    is_saturated = np.max(data,axis=-1)>threshold_sat
    local_max = np.argmax(np.multiply(data, timing_mask), axis=1)
    local_max_edge = np.argmax(np.multiply(data, timing_mask_edge), axis=1)
    ind_max_at_edge = (local_max == local_max_edge)
    local_max[ind_max_at_edge] = peak[ind_max_at_edge] - window_start
    index_max = (np.arange(0, data.shape[0]), local_max,)
    ind_with_lt_th = data[index_max] < 10.
    local_max[ind_with_lt_th] = peak[ind_with_lt_th] - window_start
    local_max[local_max < 0] = 0
    index_max = (np.arange(0, data.shape[0]), local_max,)
    charge = data[index_max]
    if np.any(is_saturated) and integration_type == 'integration_saturation': ## TODO, find a better evaluation that it is saturated
        sat_indices = tuple(np.where(is_saturated)[0])
        _data = data[sat_indices,...]
        charge[sat_indices,...] = np.apply_along_axis(contiguous_regions, 1, _data)

    return charge


def compute_gain_drop(pedestal, type='std'):
    if type == 'mean':
        return np.ones(pedestal.shape[0])
    elif type == 'std':
        return np.ones(pedestal.shape[0])
    else:
        raise('Unknown type %s' % type)


def compute_nsb_rate(pedestal, type='std'):
    if type == 'mean':
        return np.ones(pedestal.shape[0]) * 1.e9
    elif type == 'std':
        return np.ones(pedestal.shape[0]) * 1.e9
    else:
        raise('Unknown type %s' % type)





def contiguous_regions(data):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    condition = data > 0
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    val = 0.
    for start, stop in idx:
        sum_tmp = np.sum(data[start:stop])
        if val < sum_tmp: val = sum_tmp
    return val
