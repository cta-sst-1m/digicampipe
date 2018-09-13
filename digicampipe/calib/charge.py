import os

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from pkg_resources import resource_filename
from probfit import Chi2Regression
from scipy.ndimage.filters import convolve1d

from digicampipe.utils.pulse_template import NormalizedPulseTemplate

TEMPLATE_FILENAME = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)

PULSE_TEMPLATE = NormalizedPulseTemplate.load(TEMPLATE_FILENAME)


def compute_charge(events, integral_width, shift):
    """

    :param events: a stream of events
    :param integral_width: width of the integration window
    :param shift: shift to the pulse index
    :param maximum_width: width of the region (bin size) to compute charge,
    maximum value is retained. (not implemented yet)
    :return:
    """

    # bins = np.arange(-maximum_width, maximum_width + 1, 1)

    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        pulse_mask = event.data.pulse_mask

        convolved_signal = convolve1d(
            adc_samples,
            np.ones(integral_width),
            axis=-1
        )

        charges = np.zeros(convolved_signal.shape) * np.nan
        charges[pulse_mask] = convolved_signal[
            np.roll(pulse_mask, shift, axis=1)
        ]
        event.data.reconstructed_charge = charges

        yield event


def compute_charge_with_saturation(events, integral_width,
                                   saturation_threshold=300,
                                   debug=False):
    """
    :param events: a stream of events
    :param integral_width: width of the integration window
    :param saturation_threshold: if the maximum value of the waveform is above
    this threshold the waveform charge will be treated as saturated
    (type) float, ndarray
    :param debug: for debugging purposes
    :return yield events
    """

    for count, event in enumerate(events):

        adc_samples = event.data.adc_samples
        max_value = np.max(adc_samples, axis=-1)
        saturated_pulse = max_value > saturation_threshold

        convolved_signal = convolve1d(
            adc_samples,
            np.ones(integral_width),
            axis=-1
        )

        charge = np.max(convolved_signal, axis=-1)

        if np.any(saturated_pulse):
            adc_samples = adc_samples[saturated_pulse]

            # cumulative_adc_samples = np.cumsum(adc_samples, axis=-1)
            # diff2_adc_samples = np.diff(diff_adc_samples, axis=-1)
            # start_bin[:, :-1] = (adc_samples[:, :-1] < threshold_rising)
            #  * (adc_samples[:, 1:] >= threshold_rising)
            # end_bin[:, 1:-1] = (cumulative_adc_samples[:, 1:-1]
            #  >= threshold_falling) * (diff_adc_samples[:, :-1] < 0)
            # end_bin[:, 1:-1] = end_bin[:, 1:-1]
            #  * (adc_samples[:, :-2] > 0) * (adc_samples[:, 1:-1] <=0)
            diff_adc_samples = np.diff(adc_samples, axis=-1)
            samples = np.arange(adc_samples.shape[-1])
            max_diff = np.argmax(diff_adc_samples, axis=-1)[:, None] - 1
            start_bin = (samples <= max_diff)

            min_diff = np.argmin(diff_adc_samples, axis=-1)[:, None]
            end_bin = (samples >= min_diff)

            window = start_bin + end_bin
            window = ~window
            temp = adc_samples[window]
            temp = np.sum(temp, axis=-1)
            charge[saturated_pulse] = temp

        event.data.reconstructed_charge = charge

        if debug:
            pixel = 0
            print(charge)
            print(window[0], start_bin[0], end_bin[0])

            plt.figure()
            plt.plot(adc_samples[pixel])
            plt.plot(adc_samples[window][0], marker='x')

            plt.figure()
            plt.plot(np.cumsum(adc_samples, axis=-1)[0])

            plt.figure()
            plt.plot(np.diff(adc_samples)[0])
            plt.show()

        yield event


def compute_amplitude(events):
    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        pulse_indices = event.data.pulse_mask

        charges = np.ones(adc_samples.shape) * np.nan
        charges[pulse_indices] = adc_samples[pulse_indices]
        event.data.reconstructed_amplitude = charges

        yield event


def compute_full_waveform_charge(events):
    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        charges = np.sum(adc_samples, axis=-1)
        event.data.reconstructed_charge = charges.reshape(-1, 1)

        yield event


def fit_template(events, pulse_width=(4, 5), rise_time=12,
                 template=PULSE_TEMPLATE):
    for event in events:

        adc_samples = event.data.adc_samples.copy()
        time = np.arange(adc_samples.shape[-1]) * 4
        pulse_indices = event.data.pulse_mask
        pulse_indices = np.argwhere(pulse_indices)
        pulse_indices = [tuple(arr) for arr in pulse_indices]
        amplitudes = np.zeros(adc_samples.shape) * np.nan
        times = np.zeros(adc_samples.shape) * np.nan

        plt.figure()

        for pulse_index in pulse_indices:
            left = pulse_index[-1] - int(pulse_width[0])
            left = max(0, left)
            right = pulse_index[-1] + int(pulse_width[1]) + 1
            right = min(adc_samples.shape[-1] - 1, right)

            y = adc_samples[pulse_index[0], left:right]
            t = time[left:right]

            where_baseline = np.arange(adc_samples.shape[-1])
            where_baseline = (where_baseline < left) + \
                             (where_baseline >= right)
            where_baseline = adc_samples[pulse_index[0]][where_baseline]

            baseline_0 = np.mean(where_baseline)
            baseline_std = np.std(where_baseline)
            limit_baseline = (baseline_0 - baseline_std,
                              baseline_0 + baseline_std)

            t_0 = time[pulse_index[-1]] - rise_time
            limit_t = (t_0 - 2 * 4, t_0 + 2 * 4)

            amplitude_0 = np.max(y)
            limit_amplitude = (max(np.min(y), 0), amplitude_0 * 1.2)

            chi2 = Chi2Regression(template, t, y)
            m = Minuit(chi2, t_0=t_0,
                       amplitude=amplitude_0,
                       limit_t=limit_t,
                       limit_amplitude=limit_amplitude,
                       baseline=baseline_0,
                       limit_baseline=limit_baseline,
                       print_level=0, pedantic=False)
            m.migrad()

            adc_samples[pulse_index[0]] -= template(time, **m.values)
            amplitudes[pulse_index] = m.values['amplitude']
            times[pulse_index] = m.values['t']

        event.data.reconstructed_amplitude = amplitudes
        event.data.reconstructed_time = times

        yield event


def compute_photo_electron(events, gains):
    for event in events:
        charge = event.data.reconstructed_charge

        gain_drop = event.data.gain_drop
        corrected_gains = gains * gain_drop
        pe = np.nansum(charge, axis=-1) / corrected_gains
        event.data.reconstructed_number_of_pe = pe

        yield event


def compute_sample_photo_electron(events, gain_amplitude):
    """
    :param events: a stream of events
    :param gain_amplitude: Corresponds to the pulse amplitude of 1 pe in LSB
    :return: a stream of event with event.data.sample_pe filled with
    fractional pe for each pixel and each sample. Integrating the
    fractional pe along all samples gives the charge in pe of the
    full event.
    """
    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        gain_drop = event.data.gain_drop[:, None]
        sample_pe = adc_samples / (gain_amplitude[:, None] * gain_drop)
        event.data.sample_pe = sample_pe
        yield event
