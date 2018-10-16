import numba
import numpy as np


def compute_time_from_max(events):
    bin_time = 4  # 4 ns between samples

    for event in events:
        adc_samples = event.data.adc_samples
        reconstructed_time = np.argmax(adc_samples, axis=-1) * bin_time

        new_shape = reconstructed_time.shape + (1,)
        reconstructed_time = reconstructed_time.reshape(new_shape)
        event.data.reconstructed_time = reconstructed_time

        yield event


def compute_time_from_leading_edge(events, threshold=0.5):
    bin_time = 4  # 4 ns between samples

    for event in events:
        adc_samples = event.data.adc_samples

        times = estimate_time_from_leading_edge(adc_samples, threshold)
        times = times * bin_time
        new_shape = times.shape + (1,)
        times = times.reshape(new_shape)
        event.data.reconstructed_time = times

        yield event


@numba.jit
def estimate_time_from_leading_edge(adc, thr=0.5):
    """
    estimate the pulse arrival time, defined as the time the leading edge
    crossed 50% of the maximal height,
    estimated using a simple linear interpolation.

    *Note*
    This method breaks down for very small pulses where noise affects the
    leading edge significantly.
    Typical pixels have a ~2LSB electronics noise level.
    Assuming a leading edge length of 4 samples
    then for a typical pixel, pulses of 40LSB (roughly 7 p.e.)
    should be fine.

    adc: (1296, 50) dtype=uint16 or so
    thr: threshold, 50% by default ... can be played with.

    return:
        arrival_time (1296) in units of time_slices
    """
    n_pixel = adc.shape[0]
    arrival_times = np.zeros(n_pixel, dtype='f4')

    for pixel_id in range(n_pixel):
        y = adc[pixel_id]
        y -= y.min()
        am = y.argmax()
        y_ = y[:am + 1]
        lim = y_[-1] * thr
        foo = np.where(y_ < lim)[0]
        if len(foo):
            start = foo[-1]
            stop = start + 1
            arrival_times[pixel_id] = start + (
                (lim - y_[start]) / (y_[stop] - y_[start])
            )
        else:
            arrival_times[pixel_id] = np.nan
    return arrival_times
