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
from iminuit import Minuit, describe
from probfit import Chi2Regression

from histogram.histogram import Histogram1D
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.scripts import timing
from digicampipe.scripts import mpe
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.pdf import fmpe_pdf_10


def compute_data_bounds(x, y, y_err, estimated_gain, n_peaks=10,
                        params=None):

    mask = (y > 0) * (x < n_peaks * estimated_gain)

    if params is not None:

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

    x = x[mask]
    y = y[mask]
    y_err = y_err[mask]

    return x, y, y_err


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

            limit_params['limit_{}'.format(key)] = (0.5 * val, val * 1.5)

    return limit_params


def compute_init_fmpe(x, y, y_err, n_pe_peaks, snr=3, min_dist=5, debug=False):

    y = y.astype(np.float)
    cleaned_y = np.convolve(y, np.ones(min_dist), mode='same')
    cleaned_y_err = np.convolve(y_err, np.ones(min_dist), mode='same')
    bin_width = np.diff(x)
    bin_width = np.mean(bin_width)

    if (x != np.sort(x)).any():

        raise ValueError('x must be sorted !')

    d_y = np.diff(cleaned_y)
    indices = np.arange(len(y))
    peak_mask = np.zeros(y.shape, dtype=bool)
    peak_mask[1:-1] = (d_y[:-1] > 0) * (d_y[1:] < 0)
    peak_mask[1:-1] *= (cleaned_y[1:-1] / cleaned_y_err[1:-1]) > snr
    peak_indices = indices[peak_mask]
    peak_indices = peak_indices[:min(len(peak_indices), n_pe_peaks)]

    if len(peak_indices) <= 1:

        raise PeakNotFound('Not enough peak found for : \n'
                           'SNR : {} \t '
                           'Min distance : {} \n '
                           'Need a least 2 peaks, found {}!!'.
                           format(snr, min_dist, len(peak_indices)))

    x_peak = x[peak_indices]
    y_peak = y[peak_indices]
    gain = np.diff(x_peak)
    gain = np.average(gain, weights=y_peak[:-1])

    sigma = np.zeros(len(peak_indices))
    mean_peak_x = np.zeros(len(peak_indices))
    amplitudes = np.zeros(len(peak_indices))

    distance = int(gain / 2)

    if distance < bin_width:

        raise ValueError('Distance between peaks must be >= {} the bin width'
                         ''.format(bin_width))

    n_x = len(x)

    for i, peak_index in enumerate(peak_indices):

        left = x[peak_index] - distance
        left = np.searchsorted(x, left)
        left = max(0, left)
        right = x[peak_index] + distance
        right = np.searchsorted(x, right)
        right = min(n_x - 1, right)

        amplitudes[i] = np.sum(y[left:right]) * bin_width
        mean_peak_x[i] = np.average(x[left:right], weights=y[left:right])

        sigma[i] = np.average((x[left:right] - mean_peak_x[i])**2,
                              weights=y[left:right])
        sigma[i] = np.sqrt(sigma[i] - bin_width**2 / 12)

    sigma_e = np.sqrt(sigma[0]**2)
    sigma_s = (sigma[1:] ** 2 - sigma_e**2) / np.arange(1, len(sigma), 1)
    sigma_s = np.mean(sigma_s)
    sigma_s = np.sqrt(sigma_s)

    params = {'baseline': mean_peak_x[0], 'sigma_e': sigma_e,
              'sigma_s': sigma_s, 'gain': gain}

    for i, amplitude in enumerate(amplitudes):

        params['a_{}'.format(i)] = amplitude

    if debug:

        x_fit = np.linspace(np.min(x), np.max(x), num=len(x)*10)

        plt.figure()
        plt.step(x, y, where='mid', color='k', label='data')
        plt.errorbar(x, y, y_err, linestyle='None', color='k')
        plt.plot(x[peak_indices], y[peak_indices], linestyle='None',
                 marker='o', color='r', label='Peak positions')
        plt.plot(x_fit, fmpe_pdf_10(x_fit, **params), label='init', color='g')
        plt.legend(loc='best')
        plt.show()

    return params


def fit_fmpe(x, y, y_err, init_params, limit_params):

    chi2 = Chi2Regression(fmpe_pdf_10, x=x, y=y, error=y_err)

    bin_width = np.diff(x)
    bin_width = np.mean(bin_width)
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
        bin_width=bin_width,
        print_level=0,
        pedantic=False,
    )
    m.migrad()

    return m


def plot_fmpe_fit(x, y, y_err, fitter, pixel_id=None):

    if pixel_id is None:

        pixel_id = ''

    else:

        pixel_id = str(pixel_id)

    m = fitter
    n_free_parameters = len(m.list_of_vary_param())
    n_dof = len(x) - n_free_parameters

    n_events = int(np.sum(y))
    # m.draw_contour('sigma_e', 'sigma_s', show_sigma=True, bound=5)

    x_fit = np.linspace(np.min(x), np.max(x), num=len(x) * 10)
    y_fit = fmpe_pdf_10(x_fit, **m.values)

    text = '$\chi^2 / ndof : $ {:.01f} / {}\n'.format(m.fval, n_dof)
    text += 'Baseline : {:.02f} $\pm$ {:.02f} [LSB]\n'.format(
        m.values['baseline'], m.errors['baseline'])
    text += 'Gain : {:.02f} $\pm$ {:.02f} [LSB]\n'.format(m.values['gain'],
                                                            m.errors['gain'])
    text += '$\sigma_e$ : {:.02f} $\pm$ {:.02f} [LSB]\n'.format(
        m.values['sigma_e'], m.errors['sigma_e'])
    text += '$\sigma_s$ : {:.02f} $\pm$ {:.02f} [LSB]'.format(
        m.values['sigma_s'], m.errors['sigma_s'])

    data_text = r'$N_{events}$' + ' : {}\nPixel : {}'.format(n_events,
                                                             pixel_id)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.3, 0.8, 0.6])
    axes_residual = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    axes.step(x, y, where='mid', color='k', label=data_text)
    axes.errorbar(x, y, y_err, linestyle='None', color='k')
    axes.plot(x_fit, y_fit, label=text, color='r')

    y_fit = fmpe_pdf_10(x, **m.values)
    axes_residual.errorbar(x, ((y - y_fit) / y_err), marker='o', ls='None',
                           color='k')
    axes_residual.set_xlabel('[LSB]')
    axes.set_ylabel('count')
    axes_residual.set_ylabel('pull')
    # axes_residual.set_yscale('log')
    axes.legend(loc='best')

    return fig


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

        n_pe_peaks = 10
        estimated_gain = 22

        results_filename = os.path.join(output_path, 'fmpe_results.npz')

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels,
                             desc='Pixel'):

            x = charge_histo._bin_centers()
            y = charge_histo.data[i]
            y_err = charge_histo.errors()[i]

            x, y, y_err = compute_data_bounds(x, y, y_err,
                                              estimated_gain, n_pe_peaks)

            try:

                params_init = compute_init_fmpe(x, y, y_err, snr=3,
                                                min_dist=5,
                                                n_pe_peaks=n_pe_peaks,
                                                debug=debug)

                x, y, y_err = compute_data_bounds(x, y, y_err,
                                                  estimated_gain, n_pe_peaks,
                                                  params=params_init)

                params_limit = compute_limit_fmpe(params_init)

                m = fit_fmpe(x, y, y_err,
                             params_init,
                             params_limit)

                gain[i] = m.values['gain']
                gain_error[i] = m.errors['gain']
                sigma_e[i] = m.values['sigma_e']
                sigma_e_error[i] = m.errors['sigma_e']
                sigma_s[i] = m.values['sigma_s']
                sigma_s_error[i] = m.errors['sigma_s']
                baseline[i] = m.values['baseline']
                baseline_error[i] = m.errors['baseline']
                chi_2[i] = m.fval
                ndf[i] = len(x) - len(m.list_of_vary_param())

                fig = plot_fmpe_fit(x, y, y_err, m, pixel)

                fig.savefig(os.path.join(output_path, 'figures/') +
                            'fmpe_pixel_{}'.format(pixel))

                if debug:

                    plt.show()

                plt.close()

            except Exception as exception:

                print('Could not fit FMPE in pixel {}'.format(pixel))
                print(exception)

        np.savez(results_filename,
                 gain=gain, sigma_e=sigma_e,
                 sigma_s=sigma_s, baseline=baseline,
                 gain_error=gain_error, sigma_e_error=sigma_e_error,
                 sigma_s_error=sigma_s_error, baseline_error=baseline_error,
                 chi_2=chi_2, ndf=ndf,
                 pixel_id=pixel_id,
                 )

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

                charge_histo.draw(index=(i,), axis=axis_1, log=True,
                                  legend=False)
                amplitude_histo.draw(index=(i,), axis=axis_2, log=True,
                                     legend=False)
                timing_histo.draw(index=(i,), axis=axis_3, log=True,
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
        timing_histo_path = os.path.join(output_path,
                                         'timing_histo_fmpe.pk')

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
