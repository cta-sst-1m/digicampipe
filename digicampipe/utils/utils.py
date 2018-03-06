import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
# Define the integration function


def filter_template(filename_pulse_shape, n_samples, dt):

    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True)
    pulse_template = interp1d(time_steps, amplitudes, kind='cubic',
                                   bounds_error=False,
                                   fill_value=0., assume_sorted=True)

    temp_times = np.arange(time_steps[0], time_steps[-1],
                           (time_steps[1] - time_steps[0])/1000)

    temp_amplitudes = pulse_template(temp_times)
    amplitudes = amplitudes / np.max(temp_amplitudes)
    pulse_template = interp1d(time_steps, amplitudes, kind='cubic',
                                   bounds_error=False,
                                   fill_value=0., assume_sorted=True)

    t = np.arange(0, n_samples * dt, dt)
    return pulse_template(t)


def integrate(data, window_width):
    """
    Simple integration function over N samples

    :param data:
    :param options:
    :return:
    """
    if window_width == 1:
        return data
    h = ndimage.convolve1d(
        data,
        np.ones(window_width, dtype=int),
        axis=-1,
        mode='constant',
        cval=-1.e8
    )
    return h[..., (window_width-1)//2:-(window_width//2)]


def extract_charge(
    data,
    timing_mask,
    timing_mask_edge,
    peak,
    window_start,
    threshold_saturation
):
    """
    Extract the charge.
       - check which pixels are saturated
       - get the local maximum within the timing mask
         and check if it is not at the edge of the mask
       - move window_start from the maximum
    :param data:
    :param timing_mask:
    :param timing_mask_edge:
    :param peak_position:
    :param options:
    :param integration_type:
    :return:
    """
    is_saturated = np.max(data, axis=-1) > threshold_saturation
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
    # TODO, find a better evaluation that it is saturated
    if np.any(is_saturated):
        sat_indices = tuple(np.where(is_saturated)[0])
        _data = data[sat_indices, ...]
        charge[sat_indices, ...] = np.apply_along_axis(contiguous_regions, 1, _data)

    return charge, index_max


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
        if val < sum_tmp:
            val = sum_tmp
    return val


def fake_timing_hist(n_samples, timing_width, central_sample):
    """
    Create a timing array based on options.central_sample
                               and options.timing_width
    :param options:
    :param n_samples:
    :return:
    """
    timing = np.zeros((1296, n_samples+1,), dtype=float)
    cs = int(central_sample)
    tw = int(timing_width)
    timing[..., cs-tw:cs+tw] = 1.
    return timing


def generate_timing_mask(window_start, window_width, peak_positions):
    """
    Generate mask arround the possible peak position
    :param peak_positions:
    :return:
    """
    peak = np.argmax(peak_positions, axis=1)
    mask = (peak_positions.T / np.sum(peak_positions, axis=1)).T > 1e-3
    mask_window = mask + np.append(
        mask[..., 1:],
        np.zeros((peak_positions.shape[0], 1), dtype=bool),
        axis=1
    ) + np.append(
        np.zeros((peak_positions.shape[0], 1), dtype=bool),
        mask[..., :-1],
        axis=1
    )
    mask_windows_edge = mask_window * ~mask
    mask_window = mask_window[..., :-1]
    mask_windows_edge = mask_windows_edge[..., :-1]
    # window_width - int(np.floor(window_width/2))+window_start
    shift = window_start
    missing = mask_window.shape[1] - (window_width - 1)
    mask_window = mask_window[..., shift:]
    missing = mask_window.shape[1] - missing
    mask_window = mask_window[..., :-missing]
    mask_windows_edge = mask_windows_edge[..., shift:]
    mask_windows_edge = mask_windows_edge[..., :-missing]
    return peak, mask_window, mask_windows_edge
