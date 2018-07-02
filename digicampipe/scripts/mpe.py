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
  --ac_levels=<DAC>           LED AC DAC level
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --save_figures              Save the plots to the OUTPUT folder
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from probfit import Chi2Regression, describe
from iminuit import Minuit

from digicampipe.io.event_stream import calibration_event_stream
from histogram.histogram import Histogram1D
from digicampipe.calib.camera.baseline import fill_digicam_baseline, \
    subtract_baseline
from digicampipe.calib.camera.peak import fill_pulse_indices
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args, convert_dac_level
from digicampipe.utils.pdf import mpe_distribution_general
from digicampipe.utils.exception import PeakNotFound


def plot_mpe_fit(x, y, y_err, fitter, pixel_id=None):

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
    y_fit = mpe_distribution_general(x_fit, **m.values)

    text = '$\chi^2 / ndof : $ {:.01f} / {} = {:.02f} \n'.format(m.fval,
                                                                n_dof,
                                                                m.fval/n_dof)
    text += 'Baseline : {:.02f} $\pm$ {:.02f} [LSB]\n'.format(
        m.values['baseline'], m.errors['baseline'])
    text += 'Gain : {:.02f} $\pm$ {:.02f} [LSB / p.e.]\n'.format(m.values['gain'],
                                                          m.errors['gain'])
    text += '$\sigma_e$ : {:.02f} $\pm$ {:.02f} [LSB]\n'.format(
        m.values['sigma_e'], m.errors['sigma_e'])
    text += '$\sigma_s$ : {:.02f} $\pm$ {:.02f} [LSB]\n'.format(
        m.values['sigma_s'], m.errors['sigma_s'])
    text += '$\mu$ : {:.02f} $\pm$ {:.02f} [p.e.]\n'.format(
        m.values['mu'], m.errors['mu'])
    text += '$\mu_{XT}$'+' : {:.02f} $\pm$ {:.02f} [p.e.]\n'.format(
        m.values['mu_xt'], m.errors['mu_xt'])

    text += '$A$'+' : {:.02f} $\pm$ {:.02f} []\n'.format(
        m.values['amplitude'], m.errors['amplitude'])

    text += '$N_{peaks}$' + ' : {:.02f} $\pm$ {:.02f} []\n'.format(
        m.values['n_peaks'], m.errors['n_peaks'])

    data_text = r'$N_{events}$' + ' : {}\nPixel : {}'.format(n_events,
                                                             pixel_id)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.3, 0.8, 0.6])
    axes_residual = fig.add_axes([0.1, 0.1, 0.8, 0.2], sharex=axes)
    axes.step(x, y, where='mid', color='k', label=data_text)
    axes.errorbar(x, y, y_err, linestyle='None', color='k')
    axes.plot(x_fit, y_fit, label=text, color='r')

    y_fit = mpe_distribution_general(x, **m.values)
    axes_residual.errorbar(x, ((y - y_fit) / y_err), marker='o', ls='None',
                           color='k')
    # axes_residual.axhline(1, linestyle='--', color='k')
    axes_residual.set_xlabel('[LSB]')
    axes.set_ylabel('count')
    axes_residual.set_ylabel('pull')
    # axes_residual.set_yscale('log')
    axes.legend(loc='best')

    return fig


def compute_init_mpe(x, y, y_err, snr=3, min_dist=5, debug=False):

    y = y.astype(np.float)
    min_dist = int(min_dist)
    cleaned_y = np.convolve(y, np.ones(min_dist), mode='same')
    cleaned_y_err = np.sqrt(cleaned_y)
    bin_width = x[y.argmax()] - x[y.argmax() - 1]

    if (x != np.sort(x)).any():

        raise ValueError('x must be sorted !')

    d_y = np.diff(cleaned_y)
    indices = np.arange(len(y))
    peak_mask = np.zeros(y.shape, dtype=bool)
    peak_mask[1:-1] = (d_y[:-1] > 0) * (d_y[1:] <= 0)
    peak_mask[1:-1] *= (cleaned_y[1:-1] / cleaned_y_err[1:-1]) > snr
    peak_mask[min_dist:] = 0
    peak_indices = indices[peak_mask]
    peak_indices = peak_indices[:max(len(peak_indices), 1)]

    if len(peak_indices) <= 1:

        raise PeakNotFound('Not enough peak found for : \n'
                           'SNR : {} \t '
                           'Min distance : {} \n'
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
        right = x[peak_index] + distance + 1
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

    amplitude = np.sum(y) * bin_width
    mu = - np.log(amplitudes[0] / amplitude)
    baseline = mean_peak_x[0]
    n_peaks = (np.max(x) - np.min(x)) // gain
    mean_x = np.average(x, weights=y) - baseline
    mu_xt = 1 - mu * gain / mean_x
    mu_xt = max(0.02, mu_xt)

    params = {'baseline': baseline, 'sigma_e': sigma_e,
              'sigma_s': sigma_s, 'gain': gain, 'amplitude': amplitude,
              'mu': mu, 'mu_xt': mu_xt, 'n_peaks': n_peaks,
              'bin_width': bin_width}

    if debug:

        x_fit = np.linspace(np.min(x), np.max(x), num=len(x)*10)

        plt.figure()
        plt.step(x, y, where='mid', color='k', label='data')
        plt.errorbar(x, y, y_err, linestyle='None', color='k')
        plt.plot(x[peak_indices], y[peak_indices], linestyle='None',
                 marker='o', color='r', label='Peak positions')
        plt.plot(x_fit, mpe_distribution_general(x_fit, **params),
                 label='init', color='g')
        plt.legend(loc='best')
        plt.show()

    return params


def compute_limit_mpe(init_params):

    limit_params = {}

    baseline = init_params['baseline']
    gain = init_params['gain']
    sigma_e = init_params['sigma_e']
    sigma_s = init_params['sigma_s']
    mu = init_params['mu']
    amplitude = init_params['amplitude']

    limit_params['limit_baseline'] = (baseline - sigma_e, baseline + sigma_e)
    limit_params['limit_gain'] = (0.5 * gain, 1.5 * gain)
    limit_params['limit_sigma_e'] = (0.5 * sigma_e, 1.5 * sigma_e)
    limit_params['limit_sigma_s'] = (0.5 * sigma_s, 1.5 * sigma_s)
    limit_params['limit_mu'] = (0.5 * mu, 1.5 * mu)
    limit_params['limit_mu_xt'] = (0, 0.5)
    limit_params['limit_amplitude'] = (0.5 * amplitude, 1.5 * amplitude)

    return limit_params


def plot_event(events, pixel_id):

    for event in events:

        event.data.plot(pixel_id=pixel_id)
        plt.show()

        yield event


def compute(files, pixel_id, max_events, pulse_indices, integral_width,
            shift, bin_width, output_path,
            charge_histo_filename='charge_histo.pk',
            amplitude_histo_filename='amplitude_histo.pk',
            save=True):

    amplitude_histo_path = os.path.join(output_path, amplitude_histo_filename)
    charge_histo_path = os.path.join(output_path, charge_histo_filename)

    if os.path.exists(charge_histo_path) and save:

        raise IOError('File {} already exists'.format(charge_histo_path))

    if os.path.exists(amplitude_histo_path) and save:

        raise IOError('File {} already exists'.format(amplitude_histo_path))

    n_pixels = len(pixel_id)

    events = calibration_event_stream(files, pixel_id=pixel_id,
                                  max_events=max_events, baseline_new=True)
    # events = compute_baseline_with_min(events)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    # events = find_pulse_with_max(events)
    events = fill_pulse_indices(events, pulse_indices)
    events = compute_charge(events, integral_width, shift)
    events = compute_amplitude(events)

    charge_histo = Histogram1D(
        data_shape=(n_pixels,),
        bin_edges=np.arange(-40 * integral_width,
                            4096 * integral_width,
                            bin_width),
        axis_name='reconstructed charge '
                  '[LSB $\cdot$ ns]'
    )

    amplitude_histo = Histogram1D(
        data_shape=(n_pixels,),
        bin_edges=np.arange(-40, 4096, 1),
        axis_name='reconstructed amplitude '
                  '[LSB]'
    )

    for event in events:
        charge_histo.fill(event.data.reconstructed_charge)
        amplitude_histo.fill(event.data.reconstructed_amplitude)

    if save:

        charge_histo.save(charge_histo_path)
        amplitude_histo.save(amplitude_histo_path)

    return amplitude_histo, charge_histo


def entry():

    args = docopt(__doc__)
    files = args['INPUT']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['OUTPUT']

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    pixel_ids = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    ac_levels = convert_dac_level(args['--ac_levels'])
    timing_histo_filename = 'timing_histo.pk'
    timing_histo_filename = os.path.join(output_path, timing_histo_filename)

    results_filename = 'mpe_fit_results.pk'
    results_filename = os.path.join(output_path, results_filename)

    timing_histo = Histogram1D.load(timing_histo_filename)

    n_pixels = len(pixel_ids)
    n_ac_levels = len(ac_levels)

    if n_ac_levels != len(files):

        raise ValueError('n_ac levels = {} != '
                         'n_files = {}'.format(n_ac_levels, len(files)))

    if args['--compute']:

        amplitude = np.zeros((n_pixels, n_ac_levels))
        charge = np.zeros((n_pixels, n_ac_levels))
        time = np.zeros((n_pixels, n_ac_levels))

        for i, (file, ac_level) in tqdm(enumerate(zip(files, ac_levels)),
                                        total=n_ac_levels, desc='DAC level',
                                        leave=False):

            charge_histo_filename = 'charge_histo_ac_level_{}.pk' \
                                    ''.format(ac_level)
            amplitude_histo_filename = 'amplitude_histo_ac_level_{}.pk' \
                                       ''.format(ac_level)

            time[:, i] = timing_histo.mode()
            pulse_indices = time[:, i] // 4

            amplitude_histo, charge_histo = compute(
                    file,
                    pixel_ids, max_events, pulse_indices, integral_width,
                    shift, bin_width, output_path,
                    charge_histo_filename=charge_histo_filename,
                    amplitude_histo_filename=amplitude_histo_filename,
                    save=True)

            amplitude[:, i] = amplitude_histo.mean()
            charge[:, i] = charge_histo.mean()

        plt.figure()
        plt.plot(amplitude[0], charge[0])
        plt.show()

        np.savez(os.path.join(output_path, 'mpe_results'),
                 amplitude=amplitude, charge=charge, time=time,
                 pixel_ids=pixel_ids, ac_levels=ac_levels)

    if args['--fit']:

        gain = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_e = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_s = np.zeros((n_ac_levels, n_pixels)) * np.nan
        baseline = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu_xt = np.zeros((n_ac_levels, n_pixels)) * np.nan

        gain_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_e_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_s_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        baseline_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu_xt_error = np.zeros((n_ac_levels, n_pixels)) * np.nan

        chi_2 = np.zeros((n_ac_levels, n_pixels)) * np.nan
        ndf = np.zeros((n_ac_levels, n_pixels)) * np.nan

        ac_limit = [np.inf]*n_pixels

        for i, ac_level in tqdm(enumerate(ac_levels),
                                        total=n_ac_levels, desc='DAC level',
                                        leave=False):

            charge_histo_filename = 'charge_histo_ac_level_{}.pk' \
                                    ''.format(ac_level)

            charge_histo_filename = os.path.join(output_path,
                                                 charge_histo_filename)

            charge_histo = Histogram1D.load(charge_histo_filename)

            n_pixels = len(charge_histo.data)
            xx = charge_histo._bin_centers()

            for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                    desc='Pixel',
                                    leave=False):

                y = charge_histo.data[pixel_id]
                y_err = charge_histo.errors()[pixel_id]

                mask = (y > 0)
                x = xx[mask]
                y = y[mask]
                y_err = y_err[mask]

                try:

                    chi2 = Chi2Regression(mpe_distribution_general,
                                          x=x, y=y, error=y_err)

                    init_params = compute_init_mpe(x, y, y_err, snr=2,
                                                   debug=debug, min_dist=10)
                    limit_params = compute_limit_mpe(init_params)

                    options = {}

                    if init_params['baseline'] > init_params['gain'] / 2:

                        if i > 0:

                            ac_limit[j] = min(i, ac_limit[j])
                            ac_limit[j] = int(ac_limit[j])

                            options['fix_baseline'] = True
                            options['fix_gain'] = True
                            options['fix_sigma_e'] = True
                            options['fix_sigma_s'] = True

                            init_params['baseline'] = np.nanmean(
                                baseline[:ac_limit[j], j])
                            init_params['gain'] = np.nanmean(gain[:ac_limit[j], j])
                            init_params['sigma_e'] = np.nanmean(
                                sigma_e[:ac_limit[j], j])
                            init_params['sigma_s'] = np.nanmean(
                                sigma_s[:ac_limit[j], j])

                        else:

                            raise ValueError('Could not initialize the fit'
                                             'with : \n Baseline = {},'
                                             ' Gain = {}'.format(
                                init_params['baseline'], init_params['gain']))

                    m = Minuit(chi2, **init_params, **limit_params, **options,
                               print_level=0,
                               fix_bin_width=True,
                               fix_n_peaks=True,
                               pedantic=False)

                    m.migrad(ncall=100)
                    # m.minos(maxcall=100)

                    if debug:

                        print(init_params)
                        print(limit_params)
                        print(m.values)
                        print(m.merrors)
                        print(m.errors)
                        plot_mpe_fit(x, y, y_err, m, pixel_id)
                        plt.show()

                    gain[i, j] = m.values['gain']
                    sigma_e[i, j] = m.values['sigma_e']
                    sigma_s[i, j] = m.values['sigma_s']
                    baseline[i, j] = m.values['baseline']
                    mu[i, j] = m.values['mu']
                    mu_xt[i, j] = m.values['mu_xt']

                    gain_error[i, j] = m.errors['gain']
                    sigma_e_error[i, j] = m.errors['sigma_e']
                    sigma_s_error[i, j] = m.errors['sigma_s']
                    baseline_error[i, j] = m.errors['baseline']
                    mu_error[i, j] = m.errors['mu']
                    mu_xt_error[i, j] = m.errors['mu_xt']

                    chi_2[i, j] = m.fval
                    ndf[i, j] = len(x) - len(m.list_of_vary_param())

                except Exception as e:

                    print(e)
                    print('Could not fit pixel {}'.format(pixel_id))

            np.savez(results_filename,
                     gain=gain, sigma_e=sigma_e,
                     sigma_s=sigma_s, baseline=baseline,
                     mu=mu, mu_xt=mu_xt,
                     gain_error=gain_error, sigma_e_error=sigma_e_error,
                     sigma_s_error=sigma_s_error,
                     baseline_error=baseline_error,
                     mu_error=mu_error, mu_xt_error=mu_xt_error,
                     chi_2=chi_2, ndf=ndf,
                     pixel_ids=pixel_ids,
                     ac_levels=ac_levels
                     )

    if args['--save_figures']:

        pass

    if args['--display']:

        amplitude_histo_path = os.path.join(output_path,
                                            'amplitude_histo_ac_level_0.pk')
        charge_histo_path = os.path.join(output_path,
                                         'charge_histo_ac_level_0.pk')

        charge_histo = Histogram1D.load(charge_histo_path)
        charge_histo.draw(index=(0,), log=False, legend=False)

        amplitude_histo = Histogram1D.load(amplitude_histo_path)
        amplitude_histo.draw(index=(0,), log=False, legend=False)
        plt.show()

        pass

    return


if __name__ == '__main__':

    entry()
