import numpy as np
from scipy.ndimage.filters import convolve1d
from digicampipe.utils.utils import get_pulse_shape
from iminuit import Minuit
import matplotlib.pyplot as plt
from probfit import Chi2Regression


def compute_charge(events, integral_width, shift, maximum_width=2):
    """

    :param events: a stream of events
    :param integral_width: width of the integration window
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


def fit_template(events, pulse_width=(4, 5), rise_time=12):

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

            chi2 = Chi2Regression(get_pulse_shape, t, y)
            m = Minuit(chi2, t=t_0,
                       amplitude=amplitude_0,
                       limit_t=limit_t,
                       limit_amplitude=limit_amplitude,
                       baseline=baseline_0,
                       limit_baseline=limit_baseline,
                       print_level=0, pedantic=False)
            m.migrad()

            adc_samples[pulse_index[0]] -= get_pulse_shape(time, **m.values)
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


def compute_fractional_photo_electron(events, zero_pe_integral, gain_integral,
                              n_sample_integral=7, n_sample=50, n_pixel=1296):
    """
    :param events: a stream of events
    :param zero_pe_integral: value corresponding to 0 pe while integrated
    adc count (baseline subtracted) over n_sample_integral
    :param gain_integral: value corresponding to how many integrated
    adc counts (baseline subtracted) over n_sample_integral corresponds to 1 pe
    :param n_sample_integral: size of the integration window used to determine
    zero_pe_integral and gain_integral.
    :param n_sample: number of samples in the the adc reading
    :return: a stream of event with event.data.reconstructed_fraction_of_pe
    filled with fractional pe for each sample and each pixel. Integrating the
    fractional pe over n_sample_integral gives the charge in pe.
    """
    assert zero_pe_integral.shape == n_pixel
    assert gain_integral.shape == n_pixel
    frac_gain = np.tile(
        gain_integral[:, None]/n_sample_integral,
        [1, n_sample]
    )
    zero_pe_frac = np.tile(
        zero_pe_integral[:, None] / n_sample_integral,
        [1, n_sample]
    )
    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        gain_drop = event.data.gain_drop
        reconstructed_fraction_of_pe = (adc_samples - zero_pe_frac) / \
                               (frac_gain * gain_drop)
        event.data.reconstructed_fraction_of_pe = reconstructed_fraction_of_pe

