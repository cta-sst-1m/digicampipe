#!/usr/bin/env python
"""
Do the Single Photoelectron anaylsis

Usage:
  digicam-spe [options] [--] <INPUT>...

Options:
  -h --help                    Show this screen.
  --max_events=N               Maximum number of events to analyse.
  --max_histo_filename=FILE    File path of the max histogram.
                               [Default: ./max_histo.pk]
  --charge_histo_filename=FILE File path of the charge histogram
                               [Default: ./charge_histo.pk]
  --raw_histo_filename=FILE    File path of the raw histogram
                               [Default: ./raw_histo.pk]
  -o OUTPUT --output=OUTPUT    Output file path to store the results.
                               [Default: ./results.fits]
  -c --compute                 Compute the data.
  -f --fit                     Fit.
  -d --display                 Display.
  -v --debug                   Enter the debug mode.
  -p --pixel=<PIXEL>           Give a list of pixel IDs.
  --shift=N                    Number of bins to shift before integrating
                               [default: 0].
  --integral_width=N           Number of bins to integrate over
                               [default: 7].
  --pulse_finder_threshold=F   Threshold of pulse finder in arbitrary units
                               [default: 2.0].
  --save_figures=PATH          Save the plots to the indicated folder.
                               Figures are not saved is set to none
                               [default: none]
  --ncall=N                    Number of calls for the fit [default: 10000]
  --estimated_gain=N           Estimated gain to fit the parameters
                               [default: 20.0]
  --n_samples=N                Number of samples per waveform

"""
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.histogram import Histogram1D
from tqdm import tqdm
import pandas as pd
import fitsio

from digicampipe.calib.baseline import fill_baseline, subtract_baseline
from digicampipe.calib.charge import compute_charge
from digicampipe.calib.peak import find_pulse_with_max, \
    find_pulse_fast
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.scripts import raw
from digicampipe.utils.fitter import MaxHistoFitter, SPEFitter
from digicampipe.utils.docopt import convert_pixel_args, \
    convert_int, convert_text


def compute_dark_rate(number_of_zeros, total_number_of_events, time):
    p_0 = number_of_zeros / total_number_of_events
    mu_poisson = - np.log(p_0)
    rate = mu_poisson / time

    return rate


def compute_max_histo(files, histo_filename, pixel_id, max_events,
                      integral_width, shift, baseline):
    n_pixels = len(pixel_id)

    if not os.path.exists(histo_filename):

        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)
        # events = compute_baseline_with_min(events)
        events = fill_baseline(events, baseline)
        events = subtract_baseline(events)
        events = find_pulse_with_max(events)
        events = compute_charge(events, integral_width, shift)
        max_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(-4095 * integral_width,
                                4095 * integral_width),
        )

        for event in events:
            max_histo.fill(event.data.reconstructed_charge)

        max_histo.save(histo_filename)

        return max_histo

    else:

        max_histo = Histogram1D.load(histo_filename)

        return max_histo


def compute_spe(files, histo_filename, pixel_id, baseline, max_events,
                integral_width, shift, pulse_finder_threshold, debug=False):
    if not os.path.exists(histo_filename):

        n_pixels = len(pixel_id)
        events = calibration_event_stream(files,
                                          max_events=max_events,
                                          pixel_id=pixel_id)

        events = fill_baseline(events, baseline)
        events = subtract_baseline(events)
        # events = find_pulse_1(events, 0.5, 20)
        # events = find_pulse_2(events, widths=[5, 6], threshold_sigma=2)
        events = find_pulse_fast(events, threshold=pulse_finder_threshold)
        # events = find_pulse_fast_2(events, threshold=pulse_finder_threshold,
        #                           min_dist=3)
        # events = find_pulse_correlate(events,
        #                               threshold=pulse_finder_threshold)
        # events = find_pulse_gaussian_filter(events,
        #                                    threshold=pulse_finder_threshold)

        # events = find_pulse_wavelets(events, widths=[4, 5, 6],
        #                             threshold_sigma=2)

        events = compute_charge(events, integral_width=integral_width,
                                shift=shift)
        # events = compute_amplitude(events)
        # events = fit_template(events)

        # events = compute_full_waveform_charge(events)

        spe_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(-4095 * 50, 4095 * 50)
        )

        for event in events:
            spe_histo.fill(event.data.reconstructed_charge)

        spe_histo.save(histo_filename)

        return spe_histo

    else:

        spe_histo = Histogram1D.load(histo_filename)

        return spe_histo


def entry():
    args = docopt(__doc__)

    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_int(args['--max_events'])

    raw_histo_filename = args['--raw_histo_filename']
    charge_histo_filename = args['--charge_histo_filename']
    max_histo_filename = args['--max_histo_filename']
    results_filename = args['--output']

    pixel_id = convert_pixel_args(args['--pixel'])
    n_pixels = len(pixel_id)

    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    pulse_finder_threshold = float(args['--pulse_finder_threshold'])

    n_samples = int(args['--n_samples'])  # TODO access this in a better way !
    estimated_gain = float(args['--estimated_gain'])
    ncall = int(args['--ncall'])

    if args['--compute']:
        raw_histo = raw.compute(files=files,
                                max_events=max_events,
                                pixel_id=pixel_id,
                                filename=raw_histo_filename)

        baseline = raw_histo.mode()

        compute_max_histo(files=files,
                          histo_filename=max_histo_filename,
                          pixel_id=pixel_id,
                          max_events=max_events,
                          integral_width=integral_width,
                          shift=shift,
                          baseline=baseline)

        compute_spe(files=files,
                    histo_filename=charge_histo_filename,
                    pixel_id=pixel_id,
                    baseline=baseline,
                    max_events=max_events,
                    integral_width=integral_width,
                    shift=shift,
                    pulse_finder_threshold=pulse_finder_threshold,
                    debug=debug)

    if args['--fit']:

        results = {'dcr': [], 'sigma_e': [], 'mu_xt': [],
                   'gain': [], 'pixels_ids': pixel_id}

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels, desc='Pixel'):

            histo = Histogram1D.load(max_histo_filename, rows=i)

            try:
                fitter = MaxHistoFitter(histo, estimated_gain, throw_nan=True)
                fitter.fit(ncall=ncall)

                n_entries = histo.data.sum()
                number_of_zeros = fitter.parameters['a_0']
                window_length = 4 * n_samples
                rate = compute_dark_rate(number_of_zeros,
                                         n_entries,
                                         window_length)
                results['sigma_e'].append(fitter.parameters['sigma_e'])
                results['dcr'].append(rate)

                if debug:
                    fitter.draw()
                    fitter.draw_init(x_label='[LSB]')
                    fitter.draw_fit(x_label='[LSB]')
                    plt.show()

            except Exception as e:

                print('Could not compute dark count rate'
                      ' in pixel {}'.format(pixel))
                print(e)

                results['sigma_e'].append(np.nan)
                results['dcr'].append(np.nan)

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels,
                             desc='Pixel'):

            histo = Histogram1D.load(charge_histo_filename, rows=i)

            try:

                fitter = SPEFitter(histo, estimated_gain, throw_nan=True)
                fitter.fit(ncall=ncall)
                params = fitter.parameters
                n_entries = params['a_1']
                n_entries += params['a_2']
                n_entries += params['a_3']
                n_entries += params['a_4']
                crosstalk = (n_entries - params['a_1']) / n_entries
                gain = params['gain']

                results['mu_xt'].append(crosstalk)
                results['gain'].append(gain)

                if debug:
                    fitter.draw()
                    fitter.draw_init(x_label='[LSB]')
                    fitter.draw_fit(x_label='[LSB]')
                    plt.show()

            except Exception as e:

                print('Could not compute gain and crosstalk'
                      ' in pixel {}'.format(pixel))
                print(e)

                results['mu_xt'].append(np.nan)
                results['gain'].append(np.nan)

        with fitsio.FITS(results_filename, 'rw') as f:

            results = {key: np.array(val) for key, val in results.items()}
            f.write(results, extname='SPE')

    save_figure = convert_text(args['--save_figures'])
    if save_figure is not None:
        output_path = save_figure
        spe_histo = Histogram1D.load(charge_histo_filename)
        raw_histo = Histogram1D.load(raw_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        figure_directory = output_path + 'figures/'

        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        histograms = [spe_histo, max_histo]
        names = ['histogram_charge/',
                 'histo_max/']

        fitters = [SPEFitter, MaxHistoFitter]

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

        spe_histo = Histogram1D.load(charge_histo_filename)
        raw_histo = Histogram1D.load(os.path.join(output_path,
                                                  raw_histo_filename))
        max_histo = Histogram1D.load(max_histo_filename)

        spe_histo.draw(index=(0,), log=True, legend=False)
        raw_histo.draw(index=(0,), log=True, legend=False)
        max_histo.draw(index=(0,), log=True, legend=False)

        try:

            data = np.load(results_filename)
            dark_count_rate = data['dcr']
            electronic_noise = data['sigma_e']
            crosstalk = data['crosstalk']
            gain = data['gain']

        except IOError as e:

            print(e)
            print('Could not find the analysis files !')

        plt.figure()
        plt.hist(dark_count_rate[np.isfinite(dark_count_rate)],
                 bins='auto')
        plt.xlabel('dark count rate [GHz]')
        plt.legend(loc='best')

        plt.figure()
        plt.hist(crosstalk[np.isfinite(crosstalk)],
                 bins='auto')
        plt.xlabel('Crosstalk []')
        plt.legend(loc='best')

        plt.figure()
        plt.hist(gain[np.isfinite(gain)],
                 bins='auto')
        plt.xlabel('Gain [LSB/p.e.]')
        plt.legend(loc='best')

        plt.figure()
        plt.hist(electronic_noise[np.isfinite(electronic_noise)],
                 bins='auto')
        plt.xlabel('$\sigma_e$ [LSB]')
        plt.legend(loc='best')

        plt.show()

    return


if __name__ == '__main__':
    entry()
