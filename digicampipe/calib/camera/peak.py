import numpy as np
import peakutils
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import convolve1d, gaussian_filter1d
from scipy.signal import correlate
from digicampipe.utils.utils import get_pulse_shape
import matplotlib.pyplot as plt


def find_pulse_1(events, threshold, min_distance):

    for count, event in enumerate(events):

        pulse_mask = np.zeros(event.adc_samples.shape, dtype=np.bool)

        for pixel_id, adc_sample in enumerate(event.data.adc_samples):

            peak_index = peakutils.indexes(adc_sample, threshold, min_distance)
            pulse_mask[pixel_id, peak_index] = True

        event.data.pulse_mask = pulse_mask

        yield event


def find_pulse_2(events, threshold_sigma, widths, **kwargs):

    for count, event in enumerate(events):

        if count == 0:

            threshold = threshold_sigma * event.histo[0].std

        adc_samples = event.data.adc_samples
        pulse_mask = np.zeros(adc_samples.shape, dtype=np.bool)

        for pixel_id, adc_sample in enumerate(adc_samples):

            peak_index = find_peaks_cwt(adc_sample, widths, **kwargs)
            peak_index = peak_index[
                adc_sample[peak_index] > threshold[pixel_id]
            ]
            pulse_mask[pixel_id, peak_index] = True

        event.data.pulse_mask = pulse_mask

        yield event


def find_pulse_3(events, threshold):
    w = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float32)
    w /= w.sum()

    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        pulse_mask = np.zeros(adc_samples.shape, dtype=np.bool)

        c = convolve1d(
            input=adc_samples,
            weights=w,
            axis=1,
            mode='constant',
        )
        pulse_mask[:, 1:-1] = (
            (c[:, :-2] <= c[:, 1:-1]) &
            (c[:, 1:-1] >= c[:, 2:]) &
            (c[:, 1:-1] > threshold)
        )
        event.data.pulse_mask = pulse_mask

        yield event


def find_pulse_4(events, threshold):

    #time = np.arange(0, 91, 1)
    #time = time * 4

    time = np.linspace(0, 91*4, num=91)
    # time = time[time < 40]
    # time = np.tile(time, (1296, 1))
    delta_t = (time[-1] - time[0]) / len(time)

    template = get_pulse_shape(time, t=0, amplitude=1, baseline=0)
    template[template < 0.1] = 0
    template = np.tile(template, (1296, 1))

    # print(time.shape)
    # template = get_pulse_shape(time, t=0, amplitude=1, baseline=0)
    template = template / np.sum(template, axis=-1)[..., np.newaxis]

    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        pulse_mask = np.zeros(adc_samples.shape, dtype=np.bool)

        # c = correlate(adc_samples, template, mode='same')
        print(template.shape)
        print(adc_samples.shape)
        # c = correlate(adc_samples, template, mode='full')
        mean = np.mean(adc_samples, axis=-1)
        std = np.std(adc_samples, axis=-1)

        adc_samples = (adc_samples - mean[..., np.newaxis]) / std[..., np.newaxis]

        # c = convolve1d(
        #    input=adc_samples,
        #    weights=template[0],
        #    axis=1,
        #    mode='reflect',)
        # print(c.shape)

        c = correlate(adc_samples, template)

        # Convert this into lag units, but still not really physical
        shift = (len(time) - 1)

        # c = c[..., shift:]

        # Convert your lags into physical units
        #offsets = -lags * distancePerLag

        print(c.shape)

        plt.figure()
        plt.plot(c[0])
        plt.plot(c[5])
        plt.plot(c[6])
        plt.plot(c[7])
        plt.plot(adc_samples[0])
        plt.plot(template[0])
        plt.show()

        print(c.shape)

        pulse_mask[:, 1:-1] = (
            (c[:, :-2] <= c[:, 1:-1]) &
            (c[:, 1:-1] >= c[:, 2:]) &
            (c[:, 1:-1] > threshold)
        )

        print(pulse_mask[0])

        event.data.pulse_mask = pulse_mask

        yield event


def find_pulse_with_max(events):

    for i, event in enumerate(events):

        adc_samples = event.data.adc_samples

        if i == 0:

            n_samples = adc_samples.shape[-1]
            bins = np.arange(n_samples)

        arg_max = np.argmax(adc_samples, axis=-1)
        pulse_mask = (bins == arg_max[..., np.newaxis])
        event.data.pulse_mask = pulse_mask

        yield event


def find_pulse_gaussian_filter(events, threshold=2, **kwargs):

    for count, event in enumerate(events):

        c = event.data.adc_samples
        sigma = np.std(c)
        c = gaussian_filter1d(c, sigma=sigma)
        pulse_mask = np.zeros(c.shape, dtype=np.bool)

        pulse_mask[:, 1:-1] = (
            (c[:, :-2] < c[:, 1:-1]) &
            (c[:, 1:-1] > c[:, 2:]) &
            (c[:, 1:-1] > threshold)
        )

        event.data.pulse_mask = pulse_mask

        yield event


def fill_pulse_indices(events, pulse_indices):

    pulse_indices = pulse_indices.astype(np.int)

    for count, event in enumerate(events):

        if count == 0:

            data_shape = event.data.adc_samples.shape
            n_pixels = data_shape[0]
            pixel_ids = np.arange(data_shape[0], dtype=np.int)
            pulse_indices = (pixel_ids, pulse_indices)

        pulse_mask = np.zeros(data_shape, dtype=np.bool)
        pulse_mask[pulse_indices] = True
        event.data.pulse_mask = pulse_mask

        yield event
