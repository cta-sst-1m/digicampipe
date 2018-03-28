#!/usr/bin/env python
'''
Do the Single Photoelectron anaylsis

Usage:
  spe.py [options] <files>...

Options:
  -h --help     Show this screen.
  -c --compute  Compute the data
  -f --fit      Fit
  -d --display  Display
  -v --debug    Enter the debug mode
'''

from docopt import docopt
from digicampipe.io.event_stream import calibration_event_stream
import numpy as np
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D
from tqdm import tqdm
import peakutils
from scipy.signal import find_peaks_cwt
from scipy import ndimage
import scipy
from iminuit import Minuit, describe
from digicampipe.io.containers_calib import CalibrationContainer
from probfit import Chi2Regression
from digicampipe.utils.pdf import gaussian, single_photoelectron_pdf
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.histogram import (
    convert_histogram_to_container,
    convert_container_to_histogram,
)
from ctapipe.io import HDF5TableWriter, HDF5TableReader
import os
from digicampipe.io.containers_calib import (
    CalibrationEventContainer,
    CalibrationHistogramContainer,
    CalibrationResultContainer,
    SPEParameters,
    SPEResultContainer,
    CalibrationContainer,
)


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

    highest_peak_range = [
        highest_peak_index + i
        for i in range(-peak_distance, peak_distance)
    ]

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


def build_raw_data_histogram(events):

    for count, event in tqdm(enumerate(events)):

        if count == 0:

            n_pixels = event.n_pixels
            adc_histo = Histogram1D(
                data_shape=(n_pixels, ),
                bin_edges=np.arange(0, 4095, 1),
                axis_name='[LSB]'
            )

        adc_histo.fill(event.data.adc_samples)

    return convert_histogram_to_container(adc_histo)


def fill_histogram(events, id, histogram):

    for event in events:

        event.histo[id] = histogram

        yield event


def fill_electronic_baseline(events):

    for event in events:

        event.data.baseline = event.histo[0].mode

        yield event


def subtract_baseline(events):

    for event in events:

        baseline = event.data.baseline

        event.data.adc_samples = event.data.adc_samples.astype(baseline.dtype)
        event.data.adc_samples -= baseline[..., np.newaxis]

        yield event


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


def compute_charge(events, integral_width):

    for count, event in enumerate(events):

        adc_samples = event.data.adc_samples
        pulse_mask = event.data.pulse_mask

        convolved_signal = ndimage.convolve1d(
            adc_samples,
            np.ones(integral_width),
            axis=-1
        )
        charges = np.ones(convolved_signal.shape) * np.nan
        charges[pulse_mask] = convolved_signal[pulse_mask]
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

    # def f(baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4):
    #     return minimiser(
    #         x,
    #         y,
    #         y_err,
    #         spe_fit_function,
    #         baseline,
    #         gain,
    #         sigma_e,
    #         sigma_s,
    #         a_1,
    #         a_2,
    #         a_3,
    #         a_4
    #     )

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

    return m.values, m.errors, params_init, param_bounds


def minimiser(x, y, y_err, f, *args):

    return np.sum(((y - f(x, *args)) / y_err)**2)


def build_spe(events, max_events):

    spe_charge = Histogram1D(
        data_shape=(1296,),
        bin_edges=np.arange(-20, 200, 1)
    )
    spe_amplitude = Histogram1D(data_shape=(1296,),
                                bin_edges=np.arange(-20, 200, 1))

    for _, event in tqdm(zip(range(max_events), events), total=max_events):

        spe_charge.fill(event.data.reconstructed_charge)
        spe_amplitude.fill(event.data.reconstructed_amplitude)

    spe_charge = convert_histogram_to_container(spe_charge)
    spe_amplitude = convert_histogram_to_container(spe_amplitude)

    return spe_charge, spe_amplitude


def save_container(container, filename, group_name, table_name):

    with HDF5TableWriter(filename, mode='a', group_name=group_name) as h5_table:
        h5_table.write(table_name, container)


def save_event_data(events, filename, group_name):

    with HDF5TableWriter(filename, mode='a', group_name=group_name) as h5_table:

        for event in events:

            h5_table.write('waveforms', event.data)

            yield event


def main(args):

    files = args['<files>']

    debug = args['--debug']

    telescope_id = 1

    max_events = 20000

    output_file = './spe_analysis.hdf5'

    if args['--compute']:

        if not os.path.exists(output_file):

            events = calibration_event_stream(files,
                                              telescope_id=telescope_id,
                                              max_events=max_events)
            raw_histo = build_raw_data_histogram(events)
            save_container(raw_histo, output_file, 'histo', 'raw_lsb')

            events = calibration_event_stream(files,
                                              telescope_id=telescope_id,
                                              max_events=max_events)
            events = fill_histogram(events, 0, raw_histo)
            events = fill_electronic_baseline(events)
            events = subtract_baseline(events)
            # events = normalize_adc_samples(events, std)
            # events = find_pulse_1(events, 0.5, 20)
            events = find_pulse_2(events, widths=[5, 6], threshold_sigma=2)
            # events = normalize_adc_samples(events, 1./std)

            events = compute_charge(events, integral_width=7)
            events = compute_amplitude(events)
            # events = save_event_data(events, output_file, 'data')
            spe_charge, spe_amplitude = build_spe(events, max_events)

            save_container(spe_charge, output_file, 'histo', 'spe_charge')
            save_container(spe_amplitude, output_file, 'histo', 'spe_amplitude')

        else:

            raise IOError('File {} already exists'.format(output_file))

    if args['--fit']:

        # with HDF5TableReader('spe_analysis.hdf5') as h5_table:

        #     spe_charge = h5_table.read('/histo/spe_charge',
        #                                CalibrationHistogramContainer())

        #     spe_amplitude = h5_table.read('/histo/spe_amplitude',
        #                                   CalibrationHistogramContainer())

        #     spe_charge = convert_container_to_histogram(next(spe_charge))
        #     spe_amplitude = convert_container_to_histogram(next(spe_amplitude))

        # spe = spe_charge
        spe = Histogram1D.load('temp.pk')
        spe.draw(index=(10, ), log=True)
        plt.show()

        parameters = {
            'a_1': [],
            'a_2': [],
            'a_3': [],
            'a_4': [],
            'sigma_s': [],
            'sigma_e': [],
            'gain': [],
            'baseline': [],
            'pixel_id': [],
        }
        parameters_error = []

        n_pixels = spe.data.shape[0]

        results = SPEResultContainer()

        with HDF5TableWriter('spe_fit_results.h5', 'analysis', mode='a') as h5:

            for pixel_id in tqdm(range(n_pixels)):

                try:

                    params, params_err, params_init, params_bound = fit_spe(
                        spe._bin_centers(),
                        spe.data[pixel_id],
                        spe.errors(index=pixel_id), snr=3, debug=debug)

                    for key, val in params.items():
                        parameters[key].append(val)

                    parameters['pixel_id'].append(pixel_id)

                    for key, val in params_init.items():

                        results.init[key] = val
                        results.param[key] = params[key]
                        results.param_errors[key] = params_err[key]
                        results.bound_min[key] = params_bound['limit_' + key][0]
                        results.bound_max[key] = params_bound['limit_' + key][1]

                    for key, val in results.items():

                        results[key]['pixel'] = pixel_id

                    for key, val in results.items():

                        h5.write('spe_' + key, val)

                except PeakNotFound as e:

                    print(e)
                    print('Could not fit for pixel_id : {}'.format(pixel_id))

        for key, val in parameters.items():
            parameters[key] = np.array(val)

        np.savez('spe_fit_params.npz', **parameters)

    if args['--display']:

        import pandas as pd

        parameters = pd.HDFStore('spe_fit_results.h5', mode='r')
        parameters = parameters['analysis/spe_param']
        n_entries = 0

        for i in range(1, 3):

            n_entries += parameters['a_{}'.format(i)]

        xt = (n_entries - parameters['a_1']) / n_entries
        dark_count = n_entries / (4 * 92 * 10000)

        for key, val in parameters.items():

            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.hist(val, bins='auto', log=True)
            axes.set_xlabel(key + ' []')
            axes.set_ylabel('count []')

        plt.figure()
        plt.hist(xt, bins='auto', log=True)
        plt.xlabel('XT []')

        plt.figure()
        plt.hist(dark_count, bins='auto', log=True)
        plt.xlabel('dark count rate [GHz]')

        # with HDF5TableReader('spe_analysis.hdf5') as h5_table:

        #     spe_charge = h5_table.read('/histo/spe_charge',
        #                                       CalibrationHistogramContainer())

        #     spe_amplitude = h5_table.read('/histo/spe_amplitude',
        #                                           CalibrationHistogramContainer())

        #    # raw_histo = h5_table.read('/histo/raw_lsb', CalibrationHistogramContainer())

        #     spe_charge = convert_container_to_histogram(next(spe_charge))
        #    # raw_histo = convert_container_to_histogram(next(raw_histo))
        #     spe_amplitude = convert_container_to_histogram(next(spe_amplitude))

        spe_charge = Histogram1D.load('temp_10000.pk')

        # raw_histo.draw(index=(10, ))
        spe_charge.draw(index=(10, ), log=True)
        # spe_amplitude.draw(index=(10, ))

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
