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
from iminuit import Minuit, describe
from digicampipe.io.containers import CalibrationContainer
from probfit.costfunc import Chi2Regression
from digicampipe.utils.pdf import gaussian, single_photoelectron_pdf
from digicampipe.utils.exception import PeakNotFound


def compute_gaussian_parameters_highest_peak(bins, count, snr=4, debug=False):

    temp = count.copy()
    mask = ((count / np.sqrt(count)) > snr) * (count > 0)
    temp[~mask] = 0
    peak_indices = scipy.signal.argrelmax(temp, order=4)[0]

    if not len(peak_indices) > 0:

        raise PeakNotFound('Could not detect enough peaks in the histogram\n'
                           'N_peaks found : {} \n '
                           'SNR : {} \n'.format(len(peak_indices), snr))

    if len(peak_indices) == 1:

        mask = (count > 0) * (bins < bins[peak_indices[0]])
        val = np.min(count[mask])
        peak_indices = np.insert(peak_indices, 0, val)

    x_peaks = np.array(bins[peak_indices])
    # y_peaks = np.array(count[peak_indices])

    peak_distance = np.diff(x_peaks)
    peak_distance = np.mean(peak_distance) // 2
    peak_distance = peak_distance.astype(np.int)

    highest_peak_index = np.argmax(count)

    highest_peak_range = [highest_peak_index + i for i in range(-peak_distance, peak_distance)]

    bins = bins[highest_peak_range]
    count = count[highest_peak_range]

    mask = count > 0
    bins = bins[mask]
    count = count[mask]

    parameter_names = describe(gaussian)
    del(parameter_names[0])

    mean = np.average(bins, weights=count)
    std = np.average((bins - mean)**2, weights=count)
    std = np.sqrt(std)
    amplitude = np.sum(count)
    parameter_init = [mean, std, amplitude]
    parameter_init = dict(zip(parameter_names, parameter_init))

    bound_names = []

    for name in parameter_names:

        bound_names.append('limit_' + name)

    bounds = [(0, np.max(bins)),
              (0.5 * std, 1.5 * std),
              (0.5 * amplitude, 1.5 * amplitude)]

    bounds = dict(zip(bound_names, bounds))

    # gaussian_minimizer = lambda mean, sigma, amplitude: \
    #    minimiser(bins, count, np.sqrt(count),
    #              gaussian, mean, sigma, amplitude)

    gaussian_minimizer = Chi2Regression(gaussian, bins, count,
                                        error=np.sqrt(count))

    minuit = Minuit(gaussian_minimizer, **parameter_init, **bounds,
                    print_level=0, pedantic=False)
    minuit.migrad()

    if debug:

        plt.figure()
        plt.plot(bins, count)
        plt.plot(bins, gaussian(bins, **minuit.values))
        plt.show()

    return minuit.values, minuit.errors


def calibration_event_stream(path, telescope_id, max_events=None):

    container = CalibrationContainer()

    for event in event_stream(path, max_events=max_events):

        container.adc_samples = event.r0.tel[telescope_id].adc_samples
        container.n_pixels = container.adc_samples.shape[0]

        yield container


def compute_event_stats(events):

    for count, event in tqdm(enumerate(events)):

        if count == 0:

            n_pixels = event.n_pixels
            adc_histo = Histogram1D(data_shape=(n_pixels, ), bin_edges=np.arange(0, 4095, 1), axis_name='[LSB]')

        adc_histo.fill(event.adc_samples)

    return adc_histo


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


def compute_charge(events, integral_width):

    for count, event in enumerate(events):

        adc_samples = event.adc_samples
        pulse_mask = event.pulse_mask

        convolved_signal = ndimage.convolve1d(adc_samples, np.ones(integral_width), axis=-1)
        charges = np.ones(convolved_signal.shape) * np.nan
        charges[pulse_mask] = convolved_signal[pulse_mask]
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


def compute_fit_init_param(x, y, snr=4, sigma_e=None, debug=False):

    init_params = compute_gaussian_parameters_highest_peak(x, y, snr=snr,
                                                           debug=debug)[0]
    del(init_params['mean'], init_params['amplitude'])
    init_params['baseline'] = 0

    if sigma_e is None:

        init_params['sigma_s'] = init_params['sigma'] / 2
        init_params['sigma_e'] = init_params['sigma_s']

    else:

        init_params['sigma_s'] = init_params['sigma'] ** 2 - sigma_e ** 2
        init_params['sigma_s'] = np.sqrt(init_params['sigma_s'])
        init_params['sigma_e'] = sigma_e

    del (init_params['sigma'])

    temp = y.copy()
    mask = ((temp / np.sqrt(temp)) > snr) * (temp > 0)
    temp[~mask] = 0
    peak_indices = scipy.signal.argrelmax(temp, order=4)[0]

    if not len(peak_indices) > 1:

        raise PeakNotFound('Could not detect enough peak in the histogram \n'
                           'N_peaks : {} \n'
                           'SNR : {} \n'.format(len(peak_indices), snr))

    peaks_y = np.array(y[peak_indices])
    gain = np.array(x[peak_indices])
    gain = np.diff(gain)
    gain = np.mean(gain)

    init_params['gain'] = gain

    for i in range(1, max(min(peaks_y.shape[0], 4), 4)):

        val = 0

        if i < peaks_y.shape[0]:

            val = peaks_y[i]

        init_params['a_{}'.format(i)] = val

    return init_params


def fit_spe(x, y, y_err, snr=4, debug=False):

    params_init = compute_fit_init_param(x, y, snr=snr, debug=debug)

    # print(params_init)

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
        (0, 2 * params_init['gain']),
        (0, 2 * params_init['sigma_e']),
        (0, 2 * params_init['sigma_s']),
        (0, n_entries),
        (0, n_entries),
        (0, n_entries),
        (0, n_entries),
        ]

    param_bounds = dict(zip(keys, values))

    # f = lambda baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4: minimiser(x, y, y_err, spe_fit_function, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4)

    chi2 = Chi2Regression(single_photoelectron_pdf, x, y, y_err)
    m = Minuit(chi2, **params_init, **param_bounds, print_level=0, pedantic=False)
    m.migrad()

    '''
    try:
        m.minos()
    except RuntimeError:
        pass

    '''

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, single_photoelectron_pdf(x, **m.values))
        print(m.values, m.errors)
        plt.show()

    return m.values, m.errors


def minimiser(x, y, y_err, f, *args):

    return np.sum(((y - f(x, *args)) / y_err)**2)


def build_lsb_histo(filename):

    events = calibration_event_stream(filename, telescope_id=1)
    lsb_histo = compute_event_stats(events)

    return lsb_histo


def build_spe(filename, baseline, std):

    events = calibration_event_stream(filename, telescope_id=1)
    events = subtract_baseline(events, baseline)
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
    spe_amplitude.save('temp_amplitute.pk')

    return spe


def main(args):

    files = args['<files>']

    debug = args['--debug']

    if args['--compute']:

        lsb_histo = build_lsb_histo(files)
        lsb_histo.save('lsb_histo.pk')
        # lsb_histo = Histogram1D.load('lsb_histo.pk')

        # lsb_histo.draw(index=10)
        spe = build_spe(files, baseline=lsb_histo.mode(), std=lsb_histo.std())
        spe.save('spe.pk')

    if args['--fit']:

        spe = Histogram1D.load('temp.pk')

        # spe.draw(index=(10, ), log=True)
        # plt.show()

        parameters = {'a_1': [], 'a_2':[], 'a_3':[], 'a_4':[], 'sigma_s': [], 'sigma_e':[], 'gain':[], 'baseline':[], 'pixel_id':[]}
        parameters_error = []

        n_pixels = spe.data.shape[0]

        for pixel_id in tqdm(range(n_pixels)):

            try:

                params, params_err = fit_spe(
                    spe._bin_centers(),
                    spe.data[pixel_id],
                    spe.errors(index=pixel_id), snr=3, debug=debug)

                for key, val in params.items():
                    parameters[key].append(val)

                parameters['pixel_id'].append(pixel_id)

            except PeakNotFound as e:

                print(e)
                print('Could not fit for pixel_id : {}'.format(pixel_id))

        for key, val in parameters.items():
            parameters[key] = np.array(val)

        np.savez('spe_fit_params.npz', **parameters)

    if args['--display']:

        parameters = np.load('spe_fit_params.npz')

        mask = np.isfinite(parameters['sigma_e']) * np.isfinite(parameters['sigma_s']) * np.isfinite(parameters['gain'])

        plt.figure()
        plt.hist(parameters['sigma_e'][mask], bins='auto')

        plt.figure()
        plt.hist(parameters['gain'][mask], bins='auto')

        plt.figure()
        plt.hist(parameters['sigma_s'][mask], bins='auto')

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
