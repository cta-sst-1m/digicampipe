#!/usr/bin/env python
'''
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
                              [default: time.npz]
  --n_samples=N               Number of samples in readout window
'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from histogram.histogram import Histogram1D
from histogram.fit import HistogramFitter
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args

from digicampipe.scripts import timing
from digicampipe.scripts import mpe
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.pdf import fmpe_pdf_10


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
        gain = np.average(gain, weights=y_peak[:-1] ** 2)

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
        gain = np.average(gain, weights=amplitudes[:-1] ** 2)

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

    charge_histo_filename = 'charge_histo_fmpe.pk'
    amplitude_histo_filename = 'amplitude_histo_fmpe.pk'
    timing_histo_filename = os.path.join(output_path, args['--timing'])
    n_samples = int(args['--n_samples'])

    if args['--compute']:

        timing_histo = timing.compute(files, max_events, pixel_id,
                                      n_samples,
                                      filename=timing_histo_filename,
                                      save=True)

        pulse_indices = timing_histo.mode() // 4
        pulse_indices = pulse_indices.astype(int)

        mpe.compute(
            files,
            pixel_id, max_events, pulse_indices, integral_width,
            shift, bin_width, output_path,
            charge_histo_filename=charge_histo_filename,
            amplitude_histo_filename=amplitude_histo_filename,
            save=True)

    if args['--fit']:

        charge_histo = Histogram1D.load(
            os.path.join(output_path, charge_histo_filename))
        amplitude_histo = Histogram1D.load(
            os.path.join(output_path, amplitude_histo_filename))

        gain = np.zeros(n_pixels) * np.nan
        sigma_e = np.zeros(n_pixels) * np.nan
        sigma_s = np.zeros(n_pixels) * np.nan
        baseline = np.zeros(n_pixels) * np.nan
        gain_error = np.zeros(n_pixels) * np.nan
        sigma_e_error = np.zeros(n_pixels) * np.nan
        sigma_s_error = np.zeros(n_pixels) * np.nan
        baseline_error = np.zeros(n_pixels) * np.nan
        chi_2 = np.zeros(n_pixels) * np.nan
        ndf = np.zeros(n_pixels) * np.nan

        estimated_gains = [5, 20]
        ncall = int(args['--ncall'])

        histo_filenames = [amplitude_histo_filename, charge_histo_filename]

        for x, histos in tqdm(enumerate([amplitude_histo, charge_histo]),
                              total=2, desc='Histogram'):

            if x == 0:
                continue

            results_filename = 'results_' + os.path.splitext(histo_filenames[x]
                                                             )[0] + '.npz'
            results_filename = os.path.join(output_path, results_filename)

            estimated_gain = estimated_gains[x]

            for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels,
                                 desc='Pixel'):
                histo = histos[i]

                try:

                    fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                        throw_nan=True)
                    fitter.fit(ncall=ncall)

                    fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                        initial_parameters=fitter.parameters,
                                        throw_nan=True)
                    fitter.fit(ncall=ncall)

                    param = fitter.parameters
                    param_error = fitter.errors

                    gain[i] = param['gain']
                    gain_error[i] = param_error['gain']
                    sigma_e[i] = param['sigma_e']
                    sigma_e_error[i] = param_error['sigma_e']
                    sigma_s[i] = param['sigma_s']
                    sigma_s_error[i] = param_error['sigma_s']
                    baseline[i] = param['baseline']
                    baseline_error[i] = param_error['baseline']
                    chi_2[i] = fitter.fit_test() * fitter.ndf
                    ndf[i] = fitter.ndf

                    if debug:
                        x_label = 'Charge [LSB]'
                        label = 'Pixel {}'.format(pixel)

                        fitter.draw(x_label=x_label, label=label,
                                    legend=False)
                        fitter.draw_fit(x_label=x_label, label=label,
                                        legend=False)
                        fitter.draw_init(x_label=x_label, label=label,
                                         legend=False)

                        plt.show()

                except Exception as exception:

                    raise exception
                    print('Could not fit FMPE in pixel {}'.format(pixel))
                    print(exception)

            if not debug:
                np.savez(results_filename,
                         gain=gain, sigma_e=sigma_e,
                         sigma_s=sigma_s, baseline=baseline,
                         gain_error=gain_error, sigma_e_error=sigma_e_error,
                         sigma_s_error=sigma_s_error,
                         baseline_error=baseline_error,
                         chi_2=chi_2, ndf=ndf,
                         pixel_id=pixel_id,
                         )

    if args['--save_figures']:

        amplitude_histo_path = os.path.join(output_path,
                                            'amplitude_histo_fmpe.pk')
        charge_histo_path = os.path.join(output_path, 'charge_histo_fmpe.pk')

        charge_histo = Histogram1D.load(charge_histo_path)
        amplitude_histo = Histogram1D.load(amplitude_histo_path)

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

                charge_histo.draw(index=(i,), axis=axis_1, log=True,
                                  legend=False)
                amplitude_histo.draw(index=(i,), axis=axis_2, log=True,
                                     legend=False)
                figure_1.savefig(figure_path +
                                 'charge_fmpe_pixel_{}'.format(pixel))
                figure_2.savefig(figure_path +
                                 'amplitude_fmpe_pixel_{}'.format(pixel))
                figure_3.savefig(figure_path +
                                 'timing_fmpe_pixel_{}'.format(pixel))

            except Exception as e:

                print('Could not save pixel {} to : {} \n'.
                      format(pixel, figure_path))
                print(e)

            axis_1.clear()
            axis_2.clear()
            axis_3.clear()

    if args['--display']:
        amplitude_histo_path = os.path.join(output_path,
                                            'amplitude_histo_fmpe.pk')
        charge_histo_path = os.path.join(output_path,
                                         'charge_histo_fmpe.pk')

        charge_histo = Histogram1D.load(charge_histo_path)
        charge_histo.draw(index=(0,), log=False, legend=False)

        amplitude_histo = Histogram1D.load(amplitude_histo_path)
        amplitude_histo.draw(index=(0,), log=False, legend=False)

        plt.show()

        pass

    return


if __name__ == '__main__':
    entry()
