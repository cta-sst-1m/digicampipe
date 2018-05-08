#!/usr/bin/env python
'''
Do the Single Photoelectron anaylsis

Usage:
  spe.py [options] [OUTPUT] [INPUT ...]

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
  -i INPUT --input=INPUT.     Input files.
  -c --compute                Compute the data.
  -f --fit                    Fit.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --pulse_finder_threshold=F  threshold of pulse finder in arbitrary units
                              [default: 2.0].
  --save_figures              Save the plots to the OUTPUT folder
  --n_samples=N               Number of samples per waveform

'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import scipy
import pandas as pd

from iminuit import Minuit, describe
from probfit import Chi2Regression

from ctapipe.io import HDF5TableWriter
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.pdf import gaussian, single_photoelectron_pdf, log_spe
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.docopt import convert_pixel_args, convert_max_events_args
from digicampipe.io.containers_calib import SPEResultContainer
from histogram.histogram import Histogram1D
from digicampipe.calib.camera.baseline import fill_baseline, compute_baseline_with_min, subtract_baseline
from digicampipe.calib.camera.peak import find_pulse_with_max, find_pulse_gaussian_filter, find_pulse_1, find_pulse_wavelets, find_pulse_fast, find_pulse_correlate
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude, fit_template


def compute_dark_rate(number_of_zeros, total_number_of_events, time):

    p_0 = number_of_zeros / total_number_of_events
    rate = - np.log(p_0)
    rate /= time

    return rate


def compute_gaussian_parameters_first_peak(bins, count, snr=4, debug=False):

    temp = count.copy()
    mask = ((count / np.sqrt(count)) > snr) * (count > 0)
    temp[~mask] = 0
    peak_indices = scipy.signal.argrelmax(temp, order=4)[0]

    if not len(peak_indices) > 1:

        raise PeakNotFound('Could not detect enough peaks in the histogram\n'
                           'N_peaks found : {} \n '
                           'SNR : {} \n'.format(len(peak_indices), snr))

    x_peaks = np.array(bins[peak_indices])

    peak_distance = np.diff(x_peaks)
    peak_distance = np.mean(peak_distance) // 2
    peak_distance = peak_distance.astype(np.int)

    highest_peak_index = peak_indices[0]

    highest_peak_range = np.arange(-peak_distance, peak_distance + 1)
    highest_peak_range += highest_peak_index
    highest_peak_range[highest_peak_range < 0] = 0
    highest_peak_range[highest_peak_range >= len(count)] = len(count) + 1
    highest_peak_range = np.unique(highest_peak_range)

    bins = bins[highest_peak_range]
    count = count[highest_peak_range]

    mask = count > 0
    bins = bins[mask]
    count = count[mask]

    parameter_names = describe(gaussian)
    del parameter_names[0]

    mean = np.average(bins, weights=count)
    std = np.average((bins - mean)**2, weights=count)
    std = np.sqrt(std)
    amplitude = np.sum(count)
    parameter_init = [mean, std, amplitude]
    parameter_init = dict(zip(parameter_names, parameter_init))

    bound_names = []

    for name in parameter_names:

        bound_names.append('limit_' + name)

    bounds = [(np.min(bins), np.max(bins)),
              (0.5 * std, 1.5 * std),
              (0.5 * amplitude, 1.5 * amplitude)]

    bounds = dict(zip(bound_names, bounds))

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

    init_params = compute_gaussian_parameters_first_peak(x, y, snr=snr,
                                                         debug=debug)[0]
    del init_params['mean'], init_params['amplitude']
    init_params['baseline'] = 0

    if sigma_e is None:

        init_params['sigma_s'] = init_params['sigma'] / 2
        init_params['sigma_e'] = init_params['sigma_s']

    else:
        init_params['sigma_s'] = init_params['sigma'] ** 2 - sigma_e ** 2
        if init_params['sigma_s'] > 0:

            init_params['sigma_s'] = np.sqrt(init_params['sigma_s'])
        else:
            init_params['sigma_s'] = init_params['sigma']

        init_params['sigma_e'] = sigma_e

    del init_params['sigma']

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

    for i in range(0, max(min(peaks_y.shape[0], 4), 4)):

        val = 0

        if i < peaks_y.shape[0]:
            left = peak_indices[i] - int(gain / 2)
            left = max(0, left)
            right = peak_indices[i] + int(gain / 2)
            right = min(y.shape[0] - 1, right)

            if right == left:
                right = right + 1

            val = np.sum(y[left:right])

        init_params['a_{}'.format(i+1)] = val

    return init_params


def fit_spe(x, y, y_err, sigma_e, snr=4, debug=False):

    params_init = compute_fit_init_param(x, y, snr=snr, sigma_e=sigma_e,
                                         debug=debug)

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
        (- 0.5 * params_init['gain'], 1.5 * params_init['gain'] ),
        (0.5 * params_init['gain'], 1.5 * params_init['gain']),
        (0.5 * params_init['sigma_e'], 1.5 * params_init['sigma_e']),
        (0.5 * params_init['sigma_s'], 1.5 * params_init['sigma_s']),
        (0.5 * params_init['a_1'], 1.5 * params_init['a_1']),
        (0.5 * params_init['a_2'], 1.5 * params_init['a_2']),
        (0, params_init['a_2']),
        (0, params_init['a_2']),
        ]

    param_bounds = dict(zip(keys, values))

    chi2 = Chi2Regression(single_photoelectron_pdf, x, y, y_err)
    # chi2 = Chi2Regression(log_spe, x, np.log(y), np.log(y_err))
    m = Minuit(
        chi2,
        **params_init,
        **param_bounds,
        print_level=0,
        pedantic=False,
    )
    m.migrad(nsplit=5, ncall=30000)

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
        print(m.values, m.errors, m.fval)
        plt.show()

    return m.values, m.errors, params_init, param_bounds


def save_container(container, filename, group_name, table_name):

    with HDF5TableWriter(filename, mode='a', group_name=group_name) as h5:
        h5.write(table_name, container)


def save_event_data(events, filename, group_name):

    with HDF5TableWriter(filename, mode='a', group_name=group_name) as h5:

        for event in events:

            h5.write('waveforms', event.data)

            yield event


def plot_event(events, pixel_id):

    for event in events:

        event.data.plot(pixel_id=pixel_id)
        plt.show()

        yield event


def entry():

    args = docopt(__doc__)
    files = args['INPUT']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['OUTPUT']

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    pixel_id = convert_pixel_args(args['--pixel'])
    n_pixels = len(pixel_id)

    raw_histo_filename = output_path + 'raw_histo.pk'
    amplitude_histo_filename = output_path + 'amplitude_histo.pk'
    charge_histo_filename = output_path + 'charge_histo.pk'
    max_histo_filename = output_path + 'max_histo.pk'
    results_filename = output_path + 'fit_results.h5'
    dark_count_rate_filename = output_path + 'dark_count_rate.npz'
    crosstalk_filename = output_path + 'crosstalk.npz'
    electronic_noise_filename = output_path + 'electronic_noise.npz'

    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    pulse_finder_threshold = float(args['--pulse_finder_threshold'])

    n_samples = int(args['--n_samples']) # TODO access this in a better way !

    if args['--compute']:

        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)

        raw_histo = Histogram1D(
                            data_shape=(n_pixels,),
                            bin_edges=np.arange(0, 4096, 1),
                            axis_name='[LSB]'
                        )

        for i, event in tqdm(enumerate(events), total=max_events):

            raw_histo.fill(event.data.adc_samples)

        raw_histo.save(raw_histo_filename)
        raw_histo = Histogram1D.load(raw_histo_filename)
        baseline = raw_histo.mode()

        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)

        # events = compute_baseline_with_min(events)
        events = fill_baseline(events, baseline)
        events = subtract_baseline(events)
        events = find_pulse_with_max(events)
        events = compute_charge(events, integral_width, shift)

        max_histo = Histogram1D(
                            data_shape=(n_pixels,),
                            bin_edges=np.arange(-4096 * integral_width,
                                                4096 * integral_width),
                            axis_name='[LSB]'
                        )

        for event in tqdm(events, total=max_events):

            max_histo.fill(event.data.reconstructed_charge)

        max_histo.save(max_histo_filename)

        events = calibration_event_stream(files,
                                          max_events=max_events,
                                          pixel_id=pixel_id)

        events = fill_baseline(events, baseline)
        events = subtract_baseline(events)
        # events = find_pulse_1(events, 0.5, 20)
        # events = find_pulse_2(events, widths=[5, 6], threshold_sigma=2)
        # events = find_pulse_3(events, threshold=pulse_finder_threshold)
        events = find_pulse_correlate(events, threshold=pulse_finder_threshold)
        # events = find_pulse_gaussian_filter(events,
        #                                    threshold=pulse_finder_threshold)

        events = compute_charge(
            events,
            integral_width=integral_width,
            shift=shift
        )
        events = compute_amplitude(events)
        # events = fit_template(events)

        if debug:
            events = plot_event(events, 0)

        spe_charge = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(-4096 * integral_width, 4096 * integral_width)
        )
        spe_amplitude = Histogram1D(data_shape=(n_pixels,),
                                    bin_edges=np.arange(-4096,
                                                        4096,
                                                        1))

        for i, event in tqdm(enumerate(events), total=max_events):

            spe_charge.fill(event.data.reconstructed_charge)
            spe_amplitude.fill(event.data.reconstructed_amplitude)

        spe_charge.save(charge_histo_filename)
        spe_amplitude.save(amplitude_histo_filename)

    if args['--fit']:

        spe_charge = Histogram1D.load(charge_histo_filename)
        spe_amplitude = Histogram1D.load(amplitude_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        dark_count_rate = np.zeros(n_pixels) * np.nan
        electronic_noise = np.zeros(n_pixels) * np.nan

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels):

            x = max_histo._bin_centers()
            y = max_histo.data[i]

            n_entries = np.sum(y)

            mask = (y > 0)
            x = x[mask]
            y = y[mask]

            try:

                val, err = compute_gaussian_parameters_first_peak(x,
                                                                  y,
                                                                  snr=3,
                                                                  )

                number_of_zeros = val['amplitude']
                window_length = 4 * n_samples
                rate = compute_dark_rate(number_of_zeros,
                                         n_entries,
                                         window_length)
                dark_count_rate[pixel] = rate
                electronic_noise[pixel] = val['sigma']

            except Exception as e:

                print('Could not compute dark count rate in pixel {}'
                      .format(pixel))
                print(e)

        np.savez(dark_count_rate_filename, dcr=dark_count_rate)
        np.savez(electronic_noise_filename, electronic_noise)

        spe = spe_charge
        name = 'charge'
        crosstalk = np.zeros(n_pixels) * np.nan

        results = SPEResultContainer()

        table_name = 'analysis_' + name

        with HDF5TableWriter(results_filename, table_name, mode='w') as h5:

            for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels):

                try:

                    x = spe._bin_centers()
                    y = spe.data[i]
                    y_err = spe.errors(index=i)
                    sigma_e = electronic_noise[i]
                    sigma_e = sigma_e if not np.isnan(sigma_e) else None

                    params, params_err, params_init, params_bound = \
                        fit_spe(x, y, y_err, sigma_e=sigma_e,
                                snr=3, debug=debug)

                    for key, val in params.items():

                        setattr(results.init, key, params_init[key])
                        setattr(results.param, key, params[key])
                        setattr(results.param_errors, key, params_err[key])

                    for key, val in results.items():
                        results[key]['pixel_id'] = pixel

                    for key, val in results.items():

                        h5.write('spe_' + key, val)

                    n_entries = params['a_1']
                    n_entries += params['a_2']
                    n_entries += params['a_3']
                    n_entries += params['a_4']
                    crosstalk[pixel] = (n_entries - params['a_1']) / n_entries
                    print(crosstalk[pixel])
                    print(params['sigma_e'])

                except Exception as e:

                    print('Could not fit for pixel_id : {}'.format(pixel))
                    print(e)

            np.savez(crosstalk_filename, crosstalk)

    if args['--save_figures']:

        spe_charge = Histogram1D.load(charge_histo_filename)
        spe_amplitude = Histogram1D.load(amplitude_histo_filename)
        raw_histo = Histogram1D.load(raw_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        figure_directory = output_path + 'figures/'

        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        histograms = [spe_charge, spe_amplitude, raw_histo, max_histo]
        names = ['histogram_charge/', 'histogram_amplitude/', 'histogram_raw/',
                 'histo_max/']

        for i, histo in enumerate(histograms):

            figure = plt.figure()
            histogram_figure_directory = figure_directory + names[i]

            if not os.path.exists(histogram_figure_directory):
                os.makedirs(histogram_figure_directory)

            for j, pixel in enumerate(pixel_id):
                axis = figure.add_subplot(111)
                figure_path = histogram_figure_directory + 'pixel_{}'. \
                    format(pixel)

                try:

                    histo.draw(index=(j,), axis=axis, log=True, legend=False)
                    figure.savefig(figure_path)

                except Exception as e:

                    print('Could not save pixel {} to : {} \n'.
                          format(pixel, figure_path))
                    print(e)

                axis.remove()

    if args['--display']:

        spe_charge = Histogram1D.load(charge_histo_filename)
        spe_amplitude = Histogram1D.load(amplitude_histo_filename)
        raw_histo = Histogram1D.load(raw_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        spe_charge.draw(index=(0, ), log=True, legend=False)
        spe_amplitude.draw(index=(0, ), log=True, legend=False)
        raw_histo.draw(index=(0, ), log=True, legend=False)
        max_histo.draw(index=(0, ), log=True, legend=False)

        try:
            df = pd.HDFStore(results_filename, mode='r')
            parameters = df['analysis_charge/spe_param']
            parameters_error = df['analysis_charge/spe_param_errors']

            dark_count_rate = np.load(dark_count_rate_filename)['dcr']
            crosstalk = np.load(crosstalk_filename)['arr_0']
        except FileNotFoundError as e:

            print(e)
            print('Could not find the analysis files !')
            plt.show()
            exit()

        label = 'mean : {:2f} \n std : {:2f}'

        for key, val in parameters.items():

            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.hist(val, bins='auto', label=label.format(np.mean(val),
                                                           np.std(val)))
            axes.set_xlabel(key + ' []')
            axes.set_ylabel('count []')
            axes.legend(loc='best')

            # fig = plt.figure()
            # axes = fig.add_subplot(111)
            # axes.hist(parameters_error[key])
            # axes.set_xlabel(key + '_error' + ' []')
            # axes.set_ylabel('count []')

        crosstalk = crosstalk[np.isfinite(crosstalk)]
        dark_count_rate = dark_count_rate[np.isfinite(dark_count_rate)]

        plt.figure()
        plt.hist(crosstalk, bins='auto', label=label.format(np.mean(crosstalk),
                                                            np.std(crosstalk)))
        plt.xlabel('XT []')
        plt.legend(loc='best')

        plt.figure()
        plt.hist(dark_count_rate[np.isfinite(dark_count_rate)],
                 bins='auto',
                 label=label.format(np.mean(dark_count_rate),
                                    np.std(dark_count_rate)))
        plt.xlabel('dark count rate [GHz]')
        plt.legend(loc='best')

        plt.show()

    return


if __name__ == '__main__':

    entry()
