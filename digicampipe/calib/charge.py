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
    :return:
    """

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


def correct_voltage_drop(events, pde_func, xt_func, gain_func):

    for event in events:

        baseline_shift = event.data.baseline_shift

        pde_drop = pde_func(baseline_shift)
        xt_drop = xt_func(baseline_shift)
        gain_drop = gain_func(baseline_shift)

        scale = pde_drop * xt_drop * gain_drop
        event.data.reconstructed_number_of_pe /= scale

        yield event


def compute_dynamic_charge(events, integral_width, saturation_threshold=3000,
                           threshold_pulse=0.1, debug=False,  pulse_tail=False,
                           ):
    """
    :param events: a stream of events
    :param integral_width: width of the integration window for non-saturated
    pulses
    :param saturation_threshold: threshold corresponding to the pulse amplitude
    in unit of LSB at which the signal is considered saturated
    :param threshold_pulse: relative threshold (of pulse amplitude)
    at which the pulse area is integrated.
    :param debug: Enter the debug mode
    :param pulse_tail: Use or not the tail of the pulse for charge computation
    :return:
    """

    assert threshold_pulse <= 1

    for count, event in enumerate(events):

        if count == 0:

            n_pixels, n_samples = event.data.adc_samples.shape
            samples = np.arange(n_samples)
            samples = np.tile(samples, n_pixels).reshape((n_pixels, n_samples))

            if isinstance(threshold_pulse, float) or isinstance(threshold_pulse,
                                                                int):
                threshold_pulse = np.ones(n_pixels) * threshold_pulse

            if isinstance(saturation_threshold, float) or \
                    isinstance(saturation_threshold, int):
                saturation_threshold = np.ones(n_pixels) * saturation_threshold

            threshold_pulse = threshold_pulse * saturation_threshold
            threshold_pulse = threshold_pulse[:, None]

        adc_samples = event.data.adc_samples
        amplitude = np.max(adc_samples, axis=-1)

        trigger_bin = event.data.pulse_mask

        saturated_pulse = amplitude > saturation_threshold
        saturated = np.any(saturated_pulse)
        event.data.saturated = saturated

        max_arg = np.argmax(trigger_bin, axis=-1)
        start_bin = (samples <= (max_arg[:, None] - integral_width / 2))
        end_bin = (samples > (max_arg[:, None] + integral_width / 2))
        window = ~(start_bin + end_bin)

        if saturated:

            adc = adc_samples[saturated_pulse]
            smp = samples[saturated_pulse]
            threshold = threshold_pulse[saturated_pulse]
            trigger_sample = max_arg[saturated_pulse]

            start_point = trigger_sample - 3
            start_bin = (smp < start_point[:, None])
            start_bin = start_bin[:, :-1]

            end_point = (adc[:, :-1] >= threshold) * \
                        (adc[:, 1:] < threshold)
            end_point = np.argmax(end_point, axis=-1)[:, None]
            end_bin = (smp[..., :-1] > end_point + 1)
            win = ~(start_bin + end_bin) * (adc[:, :-1] > 0)

            if pulse_tail:
                extended_window = (smp[..., :-1] > end_point + 1) * \
                                  (adc[:, :-1] > 0)

                win = win + extended_window

            window[saturated_pulse, :-1] = win

        charge = np.sum(adc_samples * window, axis=-1)

        event.data.reconstructed_charge = charge
        event.data.reconstructed_amplitude = amplitude

        if debug:

            pixel = 0
            time = np.arange(adc_samples.shape[-1]) * 4
            window = window[pixel]

            plt.figure()
            plt.step(time, np.cumsum(adc_samples, axis=-1)[pixel])
            plt.xlabel('time [ns]')
            plt.ylabel('[LSB]')

            plt.figure()
            plt.step(time[:-1], np.diff(adc_samples, axis=-1)[pixel])
            plt.xlabel('time [ns]')
            plt.ylabel('[LSB]')

            baseline = event.data.baseline[pixel]
            wvf = adc_samples[pixel] + baseline

            fig = plt.figure()

            ax = fig.add_subplot(111)
            ax.step(time, wvf,
                    color='k', label='Waveform', where='mid')
            ax.axhline(baseline, xmax=1, label='DigiCam Baseline',
                       linestyle='--', color='k')
            ax.axhline(amplitude[pixel] + baseline)

            if threshold_pulse[pixel] <= amplitude[pixel]:
                ax.axhline(threshold_pulse[pixel] + baseline, linestyle='--',
                           color='b', label='Threshold')

            ax.axvline(max_arg[pixel] * 4, label='Trigger bin')

            ax.fill_between(time, baseline, wvf, where=window,
                            color='k', alpha=0.3, step='mid')
            plt.xlabel('time [ns]')
            plt.ylabel('[LSB]')
            plt.legend(loc='best')
            plt.show()

        yield event


def compute_number_of_pe_from_interpolator(events, charge_to_pe_function,
                                           debug=False):

    for event in events:

        charge = event.data.reconstructed_charge
        pe = charge_to_pe_function(charge)
        event.data.reconstructed_number_of_pe = pe

        if debug:
            print(charge)
            print(pe)

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
        pe = charge / corrected_gains
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


def interpolate_bad_pixels(events, geom, bad_pixels):
    n_bad = len(bad_pixels)
    n_pixel = len(geom.neighbors)
    average_matrix = np.zeros([n_bad, n_pixel])
    for i, pix in enumerate(bad_pixels):
        pix_neighbors = np.array(geom.neighbors[pix])
        bad_neighbors = np.intersect1d(pix_neighbors, bad_pixels,
                                       assume_unique=True)
        for bad_neighbor in bad_neighbors:
            pix_neighbors = pix_neighbors[pix_neighbors != bad_neighbor]
        average_matrix[i, pix_neighbors] = 1. / len(pix_neighbors)
    pixels_used = np.sum(average_matrix, axis=0) > 0
    average_matrix = average_matrix[:, pixels_used]
    for event in events:
        pe = event.data.reconstructed_number_of_pe
        pe[bad_pixels] = average_matrix.dot(pe[pixels_used])
        event.data.reconstructed_number_of_pe = pe
        yield event
