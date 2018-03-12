#!/usr/bin/env python
'''
Do the Single Photoelectron anaylsis

Usage:
  spe.py [options] <files>...

Options:
  -h --help     Show this screen.
'''

from docopt import docopt
from digicampipe.io.event_stream import event_stream
import numpy as np
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D
from tqdm import tqdm
import peakutils
from scipy.signal import find_peaks_cwt
from scipy import ndimage
from scipy.interpolate import splrep, sproot
import scipy
from iminuit import Minuit
from digicampipe.io.containers import CalibrationContainer
from digicampipe.utils.utils import filter_template


class FirstPhotoElectronPeakNotFound(Exception):
    pass


def calibration_event_stream(path, telescope_id, max_events=None):

    container = CalibrationContainer()

    for event in event_stream(path, max_events=max_events):

        container.adc_samples = event.r0.tel[telescope_id].adc_samples
        container.n_pixels = container.adc_samples.shape[0]

        yield container


def compute_event_stats(events):

    for count, event in enumerate(events):

        if count == 0:

            n_pixels = event.n_pixels
            adc_histo = Histogram1D(data_shape=(n_pixels, ), bin_edges=np.arange(0, 4095, 1), axis_name='[LSB]')

        adc_histo.fill(event.adc_samples)

    mode = adc_histo.mode()
    mean = adc_histo.mean()
    std = adc_histo.std()
    max = adc_histo.max

    return mean, std, mode, max


def subtract_baseline(events, baseline):

    for event in events:

        event.adc_samples = event.adc_samples.astype(baseline.dtype)
        event.adc_samples -= baseline[..., np.newaxis]

        yield event


def find_pulse_1(events, threshold, min_distance):

    for count, event in enumerate(events):

        pulse_mask = np.zeros(event.adc_samples.shape, dtype=np.bool)

        for pixel_id, adc_sample in enumerate(event.adc_samples):

            peak_index = peakutils.indexes(adc_sample, threshold, min_distance)
            pulse_mask[pixel_id, peak_index] = True

        event.pulse_mask = pulse_mask

        yield event


def find_pulse_2(events, threshold, widths, **kwargs):

    for count, event in enumerate(events):

        adc_samples = event.adc_samples
        pulse_mask = np.zeros(adc_samples.shape, dtype=np.bool)

        for pixel_id, adc_sample in enumerate(adc_samples):

            peak_index = find_peaks_cwt(adc_sample, widths, **kwargs)
            peak_index = peak_index[adc_sample[peak_index] > threshold[pixel_id]]
            pulse_mask[pixel_id, peak_index] = True

        event.pulse_mask = pulse_mask

        yield event


def compute_charge(events, width):

    for count, event in enumerate(events):

        adc_samples = event.adc_samples
        pulse_indices = event.pulse_mask

        convolved_signal = ndimage.convolve1d(adc_samples, np.ones(width), axis=-1)
        charges = np.ones(convolved_signal.shape) * np.nan
        charges[pulse_indices] = convolved_signal[pulse_indices]
        event.reconstructed_charge = charges

        yield event


def compute_amplitude(events):

    for count, event in enumerate(events):

        adc_samples = event.adc_samples
        pulse_indices = event.pulse_mask

        charges = np.ones(adc_samples.shape) * np.nan
        charges[pulse_indices] = adc_samples[pulse_indices]
        event.reconstructed_amplitude = charges

        yield event


def spe_fit_function(x, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4):

    amplitudes = np.array([a_1, a_2, a_3, a_4])
    N = np.arange(1, amplitudes.shape[0] + 1, 1)
    sigma = sigma_e**2 + N * sigma_s**2

    value = x - (N * gain + baseline)[..., np.newaxis]
    value = value**2
    value /= 2 * sigma[..., np.newaxis]
    temp = np.exp(-value) * (amplitudes / np.sqrt(sigma))[..., np.newaxis]
    temp = np.sum(temp, axis=0)
    temp /= np.sqrt(2 * np.pi)

    return temp


def compute_fit_init_param(x, y, snr=8, sigma_e=None):

    temp = y.copy()
    mask = ((temp / np.sqrt(temp)) > snr) * (temp > 0)
    temp[~mask] = 0
    peak_indices = scipy.signal.argrelmax(temp, order=4)[0]

    if not len(peak_indices) > 0:

        raise FirstPhotoElectronPeakNotFound('Could not detect any peak in the'
                                             'histogram')

    peaks_y = np.array(y[peak_indices])
    peaks_x = np.array(x[peak_indices])

    peaks_y = np.insert(peaks_y, 0, 0)
    peaks_x = np.insert(peaks_x, 0, 0)

    print(peaks_y, peaks_x)

    gain = np.diff(peaks_x)
    gain = np.mean(gain)

    print(gain)

    temp = np.argmax(peaks_y)
    one_pe_peak_x = peaks_x[temp]
    zero_pe_peak_x = peaks_x[temp - 1]

    baseline = zero_pe_peak_x
    print(x, zero_pe_peak_x)
    one_pe_peak_index = np.where(x == one_pe_peak_x)[0][0]
    print(one_pe_peak_index)

    x_right = (gain / 2).astype(np.int)
    print(x_right)
    one_pe_peak_region = [one_pe_peak_index + i for i in range(-x_right, x_right)]
    print(one_pe_peak_region)
    # y_max = np.max(y[one_pe_peak_region])
    # s = splrep(x[one_pe_peak_region], y[one_pe_peak_region] - y_max / 2, k=3)
    # roots = sproot(s)

    # print(roots)


    # interp = scipy.interpolate.interp1d(y[..., one_pe_peak_region], x[..., one_pe_peak_region], kind='cubic')

    # fwhm = interp(y[..., one_pe_peak_index]/2) - x[..., one_pe_peak_index]
    # fwhm = np.abs(roots[1] - roots[0])
    # sigma = fwhm / 2.355

    x_1 = x[one_pe_peak_region]
    y_1 = y[one_pe_peak_region]
    mean_x_1 = np.average(x_1, weights=y_1)
    std_x_1 = np.average((x_1 - mean_x_1)**2, weights=y_1)
    std_x_1 = np.sqrt(std_x_1)
    n_entries = np.sum(y_1)


    keys = [
        'limit_mean', 'limit_sigma', 'limit_amplitude'
    ]

    values = [
        (0, x_1.max()),
        (0, x_1.max()),
        (n_entries * 0.5, n_entries * 1.5)
    ]

    param_bounds = dict(zip(keys, values))

    params_init = {'mean': mean_x_1, 'sigma': std_x_1, 'amplitude': n_entries}

    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(x[peak_indices], y[peak_indices], linestyle='None', marker='o')
    # plt.plot(x_1, y_1, linestyle='None', marker='o')
    # plt.plot(x_1, gaussian(x_1, **params_init))
    # plt.show()

    f = lambda mean, sigma, amplitude: minimiser(x_1, y_1, np.sqrt(y_1), gaussian, mean, sigma, amplitude)
    m = Minuit(f, **params_init, **param_bounds, print_level=0, pedantic=False)
    m.migrad()

    # plt.plot(x, gaussian(x, **m.values))

    sigma = m.values['sigma']

    if sigma_e is None:

        sigma_e = sigma
        sigma_s = sigma

    else:

        sigma_s = np.sqrt(sigma**2 - sigma_e**2)

    keys = ['baseline', 'gain', 'sigma_e', 'sigma_s']
    for i in range(1, peaks_y.shape[0]):

        keys.append('a_{}'.format(i))

    values = [baseline, gain, sigma_e, sigma_s]
    values = values + peaks_y[1:].tolist() # append to list

    return dict(zip(keys, values))


def fit_spe(x, y, y_err):

    params_init = compute_fit_init_param(x, y, snr=8)

    mask = x > (params_init['baseline'] + params_init['gain'] / 2)
    mask *= y > 0

    x = x[mask]
    y = y[mask]
    y_err = y_err[mask]

    n_entries = np.sum(y)

    keys = [
        'limit_baseline', 'limit_gain', 'limit_sigma_e', 'limit_sigma_s',
        'limit_a_1', 'limit_a_2', 'limit_a_3', 'limit_a_4'
    ]

    values = [
        (0, params_init['gain']/2),
        (0, 3 * params_init['gain']),
        (0, params_init['gain']),
        (0, params_init['gain']),
        (0, n_entries),
        (0, n_entries),
        (0, n_entries),
        (0, n_entries),
        ]

    param_bounds = dict(zip(keys, values))

    f = lambda baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4: minimiser(x, y, y_err, spe_fit_function, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4)
    m = Minuit(f, **params_init, **param_bounds, print_level=0, pedantic=False)
    m.migrad()

    try:
        m.minos()
    except RuntimeError:
        pass

    # plt.plot(x, y)
    # plt.plot(x, spe_fit_function(x, **m.values))
    # plt.show()

    return m.values, m.errors


def gaussian(x, mean, sigma, amplitude):

    pdf = (x - mean)**2 / (2 * sigma**2)
    pdf = np.exp(-pdf)
    pdf /= np.sqrt(2 * np.pi) * sigma
    pdf *= amplitude

    return pdf


def minimiser(x, y, y_err, f, *args):

    return np.sum(((y - f(x, *args)) / y_err)**2)


def compute_histo(filename):

    events = calibration_event_stream(filename, telescope_id=1)
    mean, std, mode, max = compute_event_stats(events)

    events = calibration_event_stream(filename, telescope_id=1)
    events = subtract_baseline(events, mode)
    # events = normalize_adc_samples(events, std)
    # events = find_pulse_1(events, 0.5, 20)
    events = find_pulse_2(events, widths=[5, 6], threshold=2 * std)
    # events = normalize_adc_samples(events, 1./std)
    events = compute_charge(events, width=7)
    events = compute_amplitude(events)

    spe = Histogram1D(data_shape=(1296,), bin_edges=np.arange(-20, 200, 1))
    spe_amplitude = Histogram1D(data_shape=(1296,),
                                bin_edges=np.arange(-20, 200, 1))

    plt.figure()

    pixel = 10

    template_file = 'digicampipe/tests/resources/pulse_SST-1M_pixel_0.dat'

    # window_template = filter_template(template_file, 0.1)
    # window_template = window_template[window_template > 0]

    for event, i in tqdm(zip(events, range(10000)), total=10000):
        spe.fill(event.reconstructed_charge)
        spe_amplitude.fill(event.reconstructed_amplitude)
        # print(event.reconstructed_charge)

        # plt.plot(event.adc_samples[pixel])
        # plt.plot(charge_reco, linestyle='None', marker='x')
        # plt.plot(amp_reco, linestyle='None', marker='o')
        # plt.show()

    spe.save('temp.pk')
    spe.save('temp_amplitute.pk')


def main(args):

    files = args['<files>']


    # compute_histo(files)

    spe = Histogram1D.load('temp.pk')
    print(spe.data)

    spe.draw(index=(10, ), log=True)
    plt.show()

    # snr = 8
    # temp = spe.data.sum(axis=0)

    parameters = {'a_1': [], 'a_2':[], 'a_3':[], 'a_4':[], 'sigma_s': [], 'sigma_e':[], 'gain':[], 'baseline':[], 'pixel_id':[]}
    parameters_error = []

    for pixel_id in range(spe.data.shape[0]):

        try:
            params, params_err = fit_spe(
                spe._bin_centers(),
                spe.data[pixel_id],
                spe.errors(index=pixel_id))

            for key, val in params.items():
                parameters[key].append(val)

            parameters['pixel_id'].append(pixel_id)

        except FirstPhotoElectronPeakNotFound:

            print('Could not fit for pixel_id : {}'.format(pixel_id))

    plt.figure()
    plt.hist(parameters['sigma_e'], bins='auto')

    plt.figure()
    plt.hist(parameters['gain'], bins='auto')

    plt.figure()
    plt.hist(parameters['sigma_s'], bins='auto')

    plt.show()

    return


if __name__ == '__main__':

    args = docopt(__doc__)
    main(args)

    x = np.arange(0, 100, 0.1)

    baseline = 10
    gain = 5.8
    sigma_e = 0.8
    sigma_s = 0.8
    a = np.array([0.4, 0.1, 0.1, 0.4])
    a = a / a.sum()
    y = spe_fit_function(x, baseline, gain, sigma_e, sigma_s, a[0], a[1], a[2], a[3])

    plt.figure()
    plt.plot(x, y)
    plt.show()
