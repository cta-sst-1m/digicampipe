#!/usr/bin/env python
'''
Do the Multiple Photoelectron anaylsis

Usage:
  mpe.py [options] [OUTPUT] [INPUT ...]

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
  --save_figures              Save the plots to the OUTPUT folder
  --n_samples=N               Number of samples per waveform
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import peakutils
from iminuit import Minuit, describe
from probfit import Chi2Regression

from ctapipe.io import HDF5TableWriter
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.io.containers_calib import SPEResultContainer
from histogram.histogram import Histogram1D
from .spe import compute_gaussian_parameters_first_peak
from digicampipe.calib.camera.baseline import fill_baseline, \
    fill_digicam_baseline, subtract_baseline
from digicampipe.calib.camera.peak import find_pulse_with_max, find_pulse_gaussian_filter, find_pulse_1, find_pulse_wavelets, find_pulse_fast, find_pulse_correlate, fill_pulse_indices
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude, fit_template
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args, convert_dac_level
from digicampipe.scripts import timing
from digicampipe.scripts import mpe
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.pdf import fmpe_pdf_10


def compute_limit_fmpe(init_params):

    limit_params = {}

    baseline = init_params['baseline']
    gain = init_params['gain']
    sigma_e = init_params['sigma_e']
    sigma_s = init_params['sigma_s']

    limit_params['limit_baseline'] = (baseline - sigma_e, baseline + sigma_e)
    limit_params['limit_gain'] = (0.5 * gain, 1.5 * gain)
    limit_params['limit_sigma_e'] = (0.5 * sigma_e, 1.5 * sigma_e)
    limit_params['limit_sigma_s'] = (0.5 * sigma_s, 1.5 * sigma_s)

    for key, val in init_params.items():

        if key[:2] == 'a_':

            limit_params['limit_{}'.format(key)] = (0.5 * val,
                                                    val * 1.5)

    return limit_params


def compute_init_fmpe(x, y, yerr, n_pe_peaks,
                      thres=0.08, min_dist=5, debug=False):

    y = y.astype(np.float)
    bin_width = np.diff(x)
    bin_width = np.mean(bin_width)

    if (x != np.sort(x)).any():

        raise ValueError('x must be sorted !')

    peak_indices = peakutils.indexes(y, thres=thres, min_dist=min_dist)
    peak_indices = peak_indices[:min(len(peak_indices), n_pe_peaks)]

    if len(peak_indices) <= 1:

        raise PeakNotFound('Not enough peak found for : \n'
                           ' threshold : {} \t '
                           'min distance : {} \n '
                           'Need a least 2 peaks, found {}!!'.
                           format(thres, min_dist, len(peak_indices)))

    x_peak = x[peak_indices]
    y_peak = y[peak_indices]
    gain = np.diff(x_peak)
    gain = np.average(gain, weights=y_peak[:-1])

    sigma = np.zeros(len(peak_indices))
    mean_peak_x = np.zeros(len(peak_indices))
    amplitudes = np.zeros(len(peak_indices))

    distance = int(gain / 2)

    if distance < 1:

        raise ValueError('Distance between peaks must be >= 1 bin')

    n_x = len(x)

    for i, peak_index in enumerate(peak_indices):

        left = x[peak_index] - distance
        left = np.searchsorted(x, left)
        left = max(0, left)
        right = x[peak_index] + distance + 1
        right = np.searchsorted(x, right)
        right = min(n_x - 1, right)

        amplitudes[i] = np.sum(y[left:right]) * np.sqrt(2 * np.pi)
        mean_peak_x[i] = np.average(x[left:right], weights=y[left:right])

        sigma[i] = np.average((x[left:right] - mean_peak_x[i])**2,
                              weights=y[left:right])
        sigma[i] = np.sqrt(sigma[i] - bin_width**2 / 12)

    sigma_e = np.sqrt(sigma[0]**2 )
    sigma_s = (sigma[1:] ** 2 - sigma_e**2) / np.arange(1, len(sigma), 1)
    sigma_s = np.mean(sigma_s)
    sigma_s = np.sqrt(sigma_s)

    params = {'baseline': mean_peak_x[0], 'sigma_e': sigma_e,
              'sigma_s': sigma_s, 'gain': gain}

    for i, amplitude in enumerate(amplitudes):

        params['a_{}'.format(i)] = amplitude

    if debug:

        plt.figure()
        plt.plot(x, y)
        plt.plot(mean_peak_x, amplitudes)
        plt.show()

    return params


def fit_fmpe(x, y, yerr, init_params, limit_params, debug=False):

    chi2 = Chi2Regression(fmpe_pdf_10, x=x, y=y, error=yerr)

    params = describe(fmpe_pdf_10, verbose=False)[1:]
    fixed_params = {}

    for param in params:

        if param not in init_params.keys():

            fixed_params['fix_{}'.format(param)] = True

    m = Minuit(
        chi2,
        **init_params,
        **limit_params,
        **fixed_params,
        print_level=0,
        pedantic=False
    )
    m.migrad()

    if debug:

        print(m.values)
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fmpe_pdf_10(x, **init_params), label='init')
        plt.plot(x, fmpe_pdf_10(x, **m.values), label='fit')
        plt.legend()
        plt.show()

    return m.values, m.errors


def entry():

    args = docopt(__doc__)
    files = args['INPUT']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['OUTPUT']

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    pixel_id = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    n_samples = int(args['--n_samples'])  # TODO access this in a better way !

    n_pixels = len(pixel_id)

    charge_histo_filename = 'charge_histo_fmpe.pk'
    amplitude_histo_filename = 'amplitude_histo_fmpe.pk'
    timing_histo_filename = 'timing_histo_fmpe.pk'

    if args['--compute']:

        timing_histo = timing.compute(files, max_events, pixel_id,
                                      output_path,
                                      n_samples,
                                      filename=timing_histo_filename,
                                      save=True)

        pulse_indices = timing_histo.mode() // 4

        mpe.compute(
                files,
                pixel_id, max_events, pulse_indices, integral_width,
                shift, bin_width, output_path,
                charge_histo_filename=charge_histo_filename,
                amplitude_histo_filename=amplitude_histo_filename,
                save=True)

    if args['--fit']:

        charge_histo = Histogram1D.load(os.path.join(output_path, charge_histo_filename))
        amplitude_histo = Histogram1D.load(os.path.join(output_path, amplitude_histo_filename))

        gain = np.zeros(n_pixels) * np.nan
        sigma_e = np.zeros(n_pixels) * np.nan
        sigma_s = np.zeros(n_pixels) * np.nan
        baseline = np.zeros(n_pixels) * np.nan

        n_pe_peaks = 10
        estimated_gain = 22

        results_filename = os.path.join(output_path, 'fmpe_results.npz')

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels, desc='Pixel'):

            x = charge_histo._bin_centers()
            y = charge_histo.data[i]
            y_err = charge_histo.errors()[i]
            n_entries = np.sum(y)

            mask = (y > 0) * (x < n_pe_peaks * estimated_gain)
            x = x[mask]
            y = y[mask]
            y_err = y_err[mask]

            try:

                params_init = compute_init_fmpe(x, y, y_err, thres=0.05,
                                                min_dist=5,
                                                n_pe_peaks=n_pe_peaks)

                params_limit = compute_limit_fmpe(params_init)

                params, params_err = fit_fmpe(x, y, y_err, params_init,
                                              params_limit, debug=debug)

                gain[i] = params['gain']
                sigma_e[i] = params['sigma_e']
                sigma_s[i] = params['sigma_s']
                baseline[i] = params['baseline']

            except Exception as exception:

                print('Could not fit FMPE in pixel {}'.format(pixel))
                print(exception)

        np.savez(results_filename, gain=gain, sigma_e=sigma_e, sigma_s=sigma_s,
                 baseline=baseline, pixel_id=pixel_id)

    if args['--save_figures']:

        amplitude_histo_path = os.path.join(output_path,
                                            'amplitude_histo_fmpe.pk')
        charge_histo_path = os.path.join(output_path, 'charge_histo_fmpe.pk')
        timing_histo_path = os.path.join(output_path, 'timing_histo_fmpe.pk')

        charge_histo = Histogram1D.load(charge_histo_path)
        amplitude_histo = Histogram1D.load(amplitude_histo_path)
        timing_histo = Histogram1D.load(timing_histo_path)

        figure_path = os.path.join(output_path, 'figures/')

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        figure_1 = plt.figure()
        figure_2 = plt.figure()
        figure_3 = plt.figure()
        axis_1 = figure_1.add_subplot(111)
        axis_2 = figure_2.add_subplot(111)
        axis_3 = figure_3.add_subplot(111)

        for i, pixel in tqdm(enumerate(pixel_id), total=len(pixel_id)):

            try:

                charge_histo.draw(index=(i,), axis=axis_1, log=True, legend=False)
                amplitude_histo.draw(index=(i,), axis=axis_2, log=True, legend=False)
                timing_histo.draw(index=(i,), axis=axis_3, log=True, legend=False)
                figure_1.savefig(figure_path + 'charge_fmpe_pixel_{}'.format(pixel))
                figure_2.savefig(figure_path + 'amplitude_fmpe_pixel_{}'.format(pixel))
                figure_3.savefig(figure_path + 'timing_fmpe_pixel_{}'.format(pixel))

            except Exception as e:

                print('Could not save pixel {} to : {} \n'.
                      format(pixel, figure_path))
                print(e)

            axis_1.clear()
            axis_2.clear()
            axis_3.clear()

    if args['--display']:

        amplitude_histo_path = os.path.join(output_path, 'amplitude_histo_fmpe.pk')
        charge_histo_path = os.path.join(output_path, 'charge_histo_fmpe.pk')
        timing_histo_path = os.path.join(output_path, 'timing_histo_fmpe.pk')

        charge_histo = Histogram1D.load(charge_histo_path)
        charge_histo.draw(index=(0,), log=False, legend=False)

        amplitude_histo = Histogram1D.load(amplitude_histo_path)
        amplitude_histo.draw(index=(0,), log=False, legend=False)

        timing_histo = Histogram1D.load(timing_histo_path)
        timing_histo.draw(index=(0,), log=False, legend=False)
        plt.show()

        pass

    return


if __name__ == '__main__':

    entry()
