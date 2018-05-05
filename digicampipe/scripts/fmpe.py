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

from ctapipe.io import HDF5TableWriter
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.io.containers_calib import SPEResultContainer
from histogram.histogram import Histogram1D
from .spe import compute_gaussian_parameters_first_peak
from digicampipe.calib.camera.baseline import fill_baseline, \
    fill_digicam_baseline, subtract_baseline
from digicampipe.calib.camera.peak import find_pulse_with_max, find_pulse_gaussian_filter, find_pulse_1, find_pulse_2, find_pulse_3, find_pulse_4, fill_pulse_indices
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude, fit_template
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args, convert_dac_level
from digicampipe.scripts import timing
from digicampipe.scripts import mpe



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


def compute_init_fmpe(x, y, yerr, thres=0.08, min_dist=5):

    y = y.astype(np.float)

    if (x != np.sort(x)).any():

        raise ValueError('x must be sorted !')

    peak_indices = peakutils.indexes(y, thres=thres, min_dist=min_dist)

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

        amplitudes[i] = np.sum(y[left:right])
        mean_peak_x[i] = np.average(x[left:right], weights=y[left:right])

        sigma[i] = np.average((x[left:right] - mean_peak_x[i])**2,
                              weights=y[left:right])
        sigma[i] = np.sqrt(sigma[i])

    sigma_e = sigma[0]
    sigma_s = (sigma[1:] ** 2 - sigma[0]**2) / np.arange(1, len(sigma), 1)
    sigma_s = np.mean(sigma_s)
    sigma_s = np.sqrt(sigma_s)

    params = {'baseline': mean_peak_x[0], 'sigma_e': sigma_e,
              'sigma_s': sigma_s, 'gain': gain}

    for i, amplitude in enumerate(amplitudes):

        params['a_{}'.format(i)] = amplitude

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
    n_samples = int(args['--n_samples']) # TODO access this in a better way !

    n_pixels = len(pixel_id)

    if args['--compute']:

        charge_histo_filename = 'charge_histo_fmpe.pk'
        amplitude_histo_filename = 'amplitude_histo_fmpe.pk'
        timing_histo_filename = 'timing_histo_fmpe.pk'

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

        pass

    if args['--save_figures']:

        pass

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
