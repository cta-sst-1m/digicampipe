#!/usr/bin/env python
"""
Do Full Multiple Photoelectron anaylsis

Usage:
  digicam-fmpe [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyze
  -o OUTPUT --output=OUTPUT  Folder where to store the results.
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
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
  --ncall=N                   Number of calls for the fit [default: 10000]
  --timing=PATH               Timing filename
  --n_samples=N               Number of samples in readout window
  --estimated_gain=N          Estimated gain for the fit
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.fit import HistogramFitter
from histogram.histogram import Histogram1D
from tqdm import tqdm
import pandas as pd
from iminuit.util import describe
import fitsio
from astropy.table import Table

from digicampipe.scripts import mpe
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.pdf import fmpe_pdf_10
from digicampipe.visualization.plot import plot_array_camera, plot_histo


class FMPEFitter(HistogramFitter):
    def __init__(self, histogram, estimated_gain, n_peaks=10, **kwargs):

        self.estimated_gain = estimated_gain
        self.n_peaks = n_peaks
        super(FMPEFitter, self).__init__(histogram, **kwargs)

        self.parameters_plot_name = {'baseline': '$B$', 'gain': 'G',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$',
                                     'a_0': None, 'a_1': None, 'a_2': None,
                                     'a_3': None,
                                     'a_4': None, 'a_5': None, 'a_6': None,
                                     'a_7': None, 'a_8': None, 'a_9': None,
                                     'bin_width': None}

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, a_0, a_1, a_2,
            a_3, a_4, a_5, a_6, a_7, a_8, a_9):

        params = {'baseline': baseline, 'gain': gain, 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'a_0': a_0, 'a_1': a_1, 'a_2': a_2,
                  'a_3': a_3, 'a_4': a_4, 'a_5': a_5, 'a_6': a_6, 'a_7': a_7,
                  'a_8': a_8, 'a_9': a_9, 'bin_width': 0}

        return fmpe_pdf_10(x, **params)

    def initialize_fit(self):

        y = self.count.astype(np.float)
        x = self.bin_centers
        min_dist = self.estimated_gain / 3
        min_dist = int(min_dist)

        n_peaks = self.n_peaks

        cleaned_y = np.convolve(y, np.ones(min_dist), mode='same')
        bin_width = np.diff(x)
        bin_width = np.mean(bin_width)

        if (x != np.sort(x)).any():
            raise ValueError('x must be sorted !')

        d_y = np.diff(cleaned_y)
        indices = np.arange(len(y))
        peak_mask = np.zeros(y.shape, dtype=bool)
        peak_mask[1:-1] = (d_y[:-1] > 0) * (d_y[1:] <= 0)
        peak_mask[-min_dist:] = 0
        peak_indices = indices[peak_mask]
        peak_indices = peak_indices[:min(len(peak_indices), n_peaks)]

        if len(peak_indices) <= 1:
            raise PeakNotFound('Not enough peak found for : \n'
                               'Min distance : {} \n '
                               'Need a least 2 peaks, found {}!!'.
                               format(min_dist, len(peak_indices)))

        x_peak = x[peak_indices]
        y_peak = y[peak_indices]
        gain = np.diff(x_peak)
        weights = y_peak[:-1] ** 2
        gain = np.average(gain, weights=weights)

        sigma = np.zeros(len(peak_indices))
        mean_peak_x = np.zeros(len(peak_indices))
        amplitudes = np.zeros(len(peak_indices))

        distance = int(gain / 2)

        if distance < bin_width:
            raise ValueError(
                'Distance between peaks must be >= {} the bin width'
                ''.format(bin_width))

        n_x = len(x)

        for i, peak_index in enumerate(peak_indices):
            left = x[peak_index] - distance
            left = np.searchsorted(x, left)
            left = max(0, left)
            right = x[peak_index] + distance + 1
            right = np.searchsorted(x, right)
            right = min(n_x - 1, right)

            amplitudes[i] = np.sum(y[left:right]) * bin_width
            mean_peak_x[i] = np.average(x[left:right], weights=y[left:right])

            sigma[i] = np.average((x[left:right] - mean_peak_x[i]) ** 2,
                                  weights=y[left:right])
            sigma[i] = np.sqrt(sigma[i] - bin_width ** 2 / 12)

        gain = np.diff(mean_peak_x)
        weights = None
        # weights = amplitudes[:-1] ** 2
        gain = np.average(gain, weights=weights)

        sigma_e = np.sqrt(sigma[0] ** 2)
        sigma_s = (sigma[1:] ** 2 - sigma_e ** 2) / np.arange(1, len(sigma), 1)
        sigma_s = np.mean(sigma_s)

        if sigma_s < 0:
            sigma_s = sigma_e ** 2

        sigma_s = np.sqrt(sigma_s)

        params = {'baseline': mean_peak_x[0], 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'gain': gain}

        for i in range(n_peaks):

            if i < len(amplitudes):

                value = amplitudes[i]

            else:

                value = amplitudes.min()

            params['a_{}'.format(i)] = value

        self.initial_parameters = params

        return params

    def compute_fit_boundaries(self):

        limit_params = {}
        init_params = self.initial_parameters

        baseline = init_params['baseline']
        gain = init_params['gain']
        sigma_e = init_params['sigma_e']
        sigma_s = init_params['sigma_s']

        limit_params['limit_baseline'] = (baseline - sigma_e,
                                          baseline + sigma_e)
        limit_params['limit_gain'] = (0.5 * gain, 1.5 * gain)
        limit_params['limit_sigma_e'] = (0.5 * sigma_e, 1.5 * sigma_e)
        limit_params['limit_sigma_s'] = (0.5 * sigma_s, 1.5 * sigma_s)

        for key, val in init_params.items():

            if key[:2] == 'a_':
                limit_params['limit_{}'.format(key)] = (0.5 * val, val * 1.5)

        self.boundary_parameter = limit_params

        return limit_params

    def compute_data_bounds(self):

        x = self.histogram.bin_centers
        bin_width = np.diff(self.histogram.bins)
        y = self.histogram.data
        if not self.parameters:
            params = self.initial_parameters

        else:
            params = self.parameters

        n_peaks = self.n_peaks

        mask = (y > 0) * (x < n_peaks * self.estimated_gain)

        if 'gain' in params.keys() and 'baseline' in params.keys():

            gain = params['gain']
            baseline = params['baseline']
            amplitudes = []

            for key, val in params.items():

                if key[:2] == 'a_':
                    amplitudes.append(val)

            amplitudes = np.array(amplitudes)
            amplitudes = amplitudes[amplitudes > 0]
            n_peaks = len(amplitudes)

            min_bin = baseline - gain / 2
            max_bin = baseline + gain * (n_peaks - 1)
            max_bin += gain / 2

            mask *= (x <= max_bin) * (x >= min_bin)

        return x[mask], y[mask], bin_width[mask]


def compute(files, max_events, pixel_id, n_samples, timing_filename,
            charge_histo_filename, amplitude_histo_filename, save,
            integral_width, shift, bin_width):
    pulse_indices = np.load(timing_filename)['time'] // 4

    amplitude_histo, charge_histo = mpe.compute(
        files,
        pixel_id, max_events, pulse_indices, integral_width,
        shift, bin_width,
        charge_histo_filename=charge_histo_filename,
        amplitude_histo_filename=amplitude_histo_filename,
        save=save)

    return amplitude_histo, charge_histo


def plot_results(results_filename, figure_path=None):

    fit_results = Table.read(results_filename, format='fits')
    fit_results = fit_results.to_pandas()

    gain = fit_results['gain']
    sigma_e = fit_results['sigma_e']
    sigma_s = fit_results['sigma_s']
    baseline = fit_results['baseline']

    _, fig_1 = plot_array_camera(gain, label='Gain [LSB $\cdot$ ns]')
    _, fig_2 = plot_array_camera(sigma_e, label='$\sigma_e$ [LSB $\cdot$ ns]')
    _, fig_3 = plot_array_camera(sigma_s, label='$\sigma_s$ [LSB $\cdot$ ns]')
    _, fig_7 = plot_array_camera(baseline, label='Baseline [LSB]')

    fig_4 = plot_histo(gain, x_label='Gain [LSB $\cdot$ ns]', bins='auto')
    fig_5 = plot_histo(sigma_e, x_label='$\sigma_e$ [LSB $\cdot$ ns]',
                       bins='auto')
    fig_6 = plot_histo(sigma_s, x_label='$\sigma_s$ [LSB $\cdot$ ns]',
                       bins='auto')
    fig_8 = plot_histo(baseline, x_label='Baseline [LSB]', bins='auto')

    if figure_path is not None:

        fig_1.savefig(os.path.join(figure_path, 'gain_camera'))
        fig_2.savefig(os.path.join(figure_path, 'sigma_e_camera'))
        fig_3.savefig(os.path.join(figure_path, 'sigma_s_camera'))
        fig_7.savefig(os.path.join(figure_path, 'baseline_camera'))

        fig_4.savefig(os.path.join(figure_path, 'gain_histo'))
        fig_5.savefig(os.path.join(figure_path, 'sigma_e_histo'))
        fig_6.savefig(os.path.join(figure_path, 'sigma_s_histo'))
        fig_8.savefig(os.path.join(figure_path, 'baseline_histo'))


def plot_fit(histo, results_filename, pixel, figure_path=None):

    fit_results = Table.read(results_filename, format='fits')
    fit_results = fit_results.to_pandas()
    hist = histo[pixel]
    fitter = FMPEFitter(hist, throw_nan=True,
                        estimated_gain=fit_results['gain'][pixel])

    for key in fitter.parameters_name:

        fitter.parameters[key] = fit_results[key][pixel]
        fitter.errors[key] = fit_results[key + '_error'][pixel]

    fitter.ndf = fit_results['ndf'][pixel]

    x_label = 'Charge [LSB]'
    label = 'Pixel {}'.format(pixel)
    fig = fitter.draw_fit(x_label=x_label, label=label, legend=False,
                          log=True)

    if figure_path is not None:

        figure_name = 'charge_fmpe_pixel_{}'.format(pixel)
        figure_name = os.path.join(figure_path, figure_name)
        fig.savefig(figure_name)


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['--output']

    if not os.path.exists(output_path):
        raise IOError('Path for output does not exists \n')

    pixel_id = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])

    n_pixels = len(pixel_id)

    charge_histo_filename = os.path.join(output_path, 'charge_histo_fmpe.pk')
    amplitude_histo_filename = os.path.join(output_path,
                                            'amplitude_histo_fmpe.pk')
    results_filename = os.path.join(output_path, 'fmpe_fit_results.fits')
    timing_filename = args['--timing']
    n_samples = int(args['--n_samples'])
    ncall = int(args['--ncall'])
    estimated_gain = float(args['--estimated_gain'])

    if args['--compute']:
        compute(files,
                max_events=max_events,
                pixel_id=pixel_id,
                n_samples=n_samples,
                timing_filename=timing_filename,
                charge_histo_filename=charge_histo_filename,
                amplitude_histo_filename=amplitude_histo_filename,
                save=True,
                integral_width=integral_width,
                shift=shift,
                bin_width=bin_width)

    if args['--fit']:

        charge_histo = Histogram1D.load(charge_histo_filename)

        param_names = describe(FMPEFitter.pdf)[2:]
        param_error_names = [key + '_error' for key in param_names]
        columns = param_names + param_error_names
        columns = columns + ['chi_2', 'ndf']
        data = np.zeros((n_pixels, len(columns))) * np.nan

        results = pd.DataFrame(data=data, columns=columns)

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels,
                             desc='Pixel'):
            histo = charge_histo[i]

            try:

                fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                    throw_nan=True)
                fitter.fit(ncall=ncall)

                fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                    initial_parameters=fitter.parameters,
                                    throw_nan=True)
                fitter.fit(ncall=ncall)

                param = dict(fitter.parameters)
                param_error = dict(fitter.errors)
                param_error = {key + '_error': val for key, val in
                               param_error.items()}

                param.update(param_error)
                param['chi_2'] = fitter.fit_test() * fitter.ndf
                param['ndf'] = fitter.ndf
                results.iloc[pixel] = param

                if debug:
                    x_label = 'Charge [LSB]'
                    label = 'Pixel {}'.format(pixel)

                    fitter.draw(x_label=x_label, label=label,
                                legend=False)
                    fitter.draw_fit(x_label=x_label, label=label,
                                    legend=False)
                    fitter.draw_init(x_label=x_label, label=label,
                                     legend=False)

                    print(results.iloc[pixel])

                    plt.show()

            except Exception as exception:

                print('Could not fit FMPE in pixel {}'.format(pixel))
                print(exception)

        if not debug:

            with fitsio.FITS(results_filename, 'rw') as f:

                f.write(results.to_records(index=False))

    if args['--save_figures']:

        charge_histo = Histogram1D.load(charge_histo_filename)

        figure_path = os.path.join(output_path, 'figures/')

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        plot_results(results_filename, figure_path)

        for i, pixel in tqdm(enumerate(pixel_id), total=len(pixel_id)):

            try:

                plot_fit(charge_histo, results_filename, i,
                         figure_path)

                plt.close()

            except Exception as e:

                print('Could not save pixel {} to : {} \n'.
                      format(pixel, figure_path))
                print(e)

    if args['--display']:

        pixel = 0
        charge_histo = Histogram1D.load(charge_histo_filename)

        plot_results(results_filename)
        plot_fit(charge_histo, results_filename, pixel=pixel)

        plt.show()

        pass

    return


if __name__ == '__main__':
    entry()
