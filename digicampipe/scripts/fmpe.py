#!/usr/bin/env python
'''
Do the Multiple Photoelectron anaylsis

Usage:
  fmpe.py [options] [FILE] [INPUT ...]

Options:
  -h --help               Show this screen.
  --max_events=N          Maximum number of events to analyse
  -o FILE --output=FILE.  Output file.
  -i INPUT --input=INPUT. Input files.
  -c --compute            Compute the data.
  -d --dark
  -t --time
  -m --fmpe
  -f --fit                Fit.
  -d --display            Display.
  -v --debug              Enter the debug mode.
  -p --pixel=<PIXEL>      Give a list of pixel IDs.
'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import scipy
import pandas as pd
from scipy.ndimage.filters import convolve1d

import peakutils
from iminuit import Minuit, describe
from probfit import Chi2Regression

from ctapipe.io import HDF5TableWriter, HDF5TableReader
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.pdf import gaussian, single_photoelectron_pdf
from digicampipe.utils.exception import PeakNotFound
from digicampipe.io.containers_calib import SPEResultContainer, \
    CalibrationHistogramContainer, CalibrationContainer
from histogram.histogram import Histogram1D
from digicampipe.utils.utils import get_pulse_shape
from digicampipe.scripts.spe import compute_charge, compute_amplitude, subtract_baseline
from digicampipe.utils.pdf import fmpe_pdf_10


def fill_baseline(events, baseline):

    for event in events:

        event.data.baseline = baseline

        yield event


def fill_peak_position(events, peak_position):

    peak_position = peak_position.reshape(peak_position.shape + (1, ))

    for count, event in enumerate(events):

        if count == 0:

            adc_samples = event.data.adc_samples
            pulse_mask = np.arange(adc_samples.shape[1])
            pulse_mask = np.tile(pulse_mask, (adc_samples.shape[0], 1))
            pulse_mask = (pulse_mask == peak_position)

        event.data.pulse_mask = pulse_mask

        yield event


def _convert_pixel_args(text):

    if text is not None:

        text = text.split(',')
        pixel_id = list(map(int, text))
        pixel_id = np.array(pixel_id)

    else:

        pixel_id = np.arange(0, 1296, 1)

    return pixel_id


def _convert_ac_level_args(text):

    if text is not None:

        text = text.split(',')
        ac_level = list(map(int, text))
        ac_level = np.array(ac_level)

    else:

        ac_level = None

    return ac_level


def entry():

    args = docopt(__doc__)
    input_files = args['INPUT']
    pixel_id = _convert_pixel_args(args['--pixel'])
    debug = args['--debug']

    max_events = args['--max_events']
    max_events = max_events if max_events is None else int(max_events)

    fmpe_filename = 'fmpe.pk'
    time_histo_filename = 'time.pk'
    dark_histo_filename = 'dark.pk'

    print(max_events, pixel_id, input_files)

    if args['--compute']:

        if args['--time']:

            events = calibration_event_stream(input_files, telescope_id=1,
                                          max_events=max_events,
                                          pixel_id=pixel_id)

            time_histo = Histogram1D(bin_edges=np.arange(0, 100, 1),
                                     data_shape=(len(pixel_id), ))

            for event in tqdm(events, total=max_events):

                adc_samples = event.data.adc_samples
                bin_max = np.argmax(adc_samples, axis=-1)
                bin_max = bin_max.reshape((bin_max.shape + (1, )))
                time_histo.fill(bin_max)

            time_histo.save(time_histo_filename)
            time_histo.draw(index=(0, ))

            plt.show()

        if args['--dark']:

            time_histo = Histogram1D.load(time_histo_filename)
            peak_position = time_histo.mode()

            events = calibration_event_stream(input_files, telescope_id=1,
                                              max_events=max_events,
                                              pixel_id=pixel_id)

            events = fill_peak_position(events,
                                        peak_position=peak_position)
            events = compute_amplitude(events)

            dark_histo = Histogram1D(bin_edges=np.arange(0, 4096, 1),
                                     data_shape=(len(pixel_id),))

            for event in tqdm(events, total=max_events):

                dark_histo.fill(event.data.reconstructed_amplitude)

            dark_histo.save(dark_histo_filename)
            dark_histo.draw(index=(0, ))
            plt.show()

        if args['--fmpe']:

            dark_histo = Histogram1D.load(dark_histo_filename)
            time_histo = Histogram1D.load(time_histo_filename)

            peak_position = time_histo.mode()
            baseline = dark_histo.mean()

            events = calibration_event_stream(input_files, telescope_id=1,
                                          max_events=max_events,
                                          pixel_id=pixel_id)

            events = fill_peak_position(events, peak_position=peak_position)
            events = fill_baseline(events, baseline)
            events = subtract_baseline(events)
            events = compute_charge(events, integral_width=7, shift=0)

            fmpe = Histogram1D(bin_edges=np.arange(-4096, 4096, 1),
                               data_shape=(len(pixel_id), ),
                               axis_name='[LSB]')

            for count, event in tqdm(enumerate(events), total=max_events):

                fmpe.fill(event.data.reconstructed_charge)

            fmpe.save(fmpe_filename)
            fmpe.draw(index=(0, ), log=True)
            plt.show()

    if args['--fit']:

        fmpe = Histogram1D.load(fmpe_filename)
        bin_centers = fmpe._bin_centers()

        for pixel_id, y in tqdm(enumerate(fmpe.data),
                                total=fmpe.data.shape[0]):
            mask = y > 0
            y = y[mask]
            x = bin_centers[mask]
            yerr = np.sqrt(y)

            init_params = compute_init_fmpe(x, y, yerr)
            limit_params = compute_limit_fmpe(init_params)

            fit_fmpe(x, y, yerr, init_params, limit_params, debug)


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
                                                    val * 1.5 )

    return limit_params


def compute_init_fmpe(x, y, yerr, thres=0.08, min_dist=5):

    y = y.astype(np.float)

    if (x != np.sort(x)).any():

        raise ValueError('x must be sorted !')

    peak_indices = peakutils.indexes(y, thres=thres, min_dist=min_dist)

    if len(peak_indices) <= 1:

       raise PeakNotFound('Not enough peak found for : \n '
                          'threshold : {} \t'
                          'min distance : {} \n'
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


if __name__ == '__main__':
    entry()
