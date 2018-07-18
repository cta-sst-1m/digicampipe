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
  --params=<YAML_FILE>        Calibration params to use in the fit
'''
import os
from docopt import docopt
from tqdm import tqdm
import yaml

import numpy as np
import matplotlib.pyplot as plt
from probfit import Chi2Regression, describe
from iminuit import Minuit

from histogram.histogram import Histogram1D
from histogram.fit import HistogramFitter

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, \
    subtract_baseline
from digicampipe.calib.camera.peak import fill_pulse_indices
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args, convert_dac_level
from digicampipe.utils.pdf import mpe_distribution_general


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

    return fig, axes


def compute_init_mpe(x, y, y_err, fixed_params, debug=False):

    bin_width = fixed_params['bin_width']
    gain = fixed_params['gain']
    sigma_e = fixed_params['sigma_e']
    sigma_s = fixed_params['sigma_s']
    baseline = fixed_params['baseline']

    mean_x = np.average(x, weights=y) - baseline

    if 'mu_xt' in fixed_params.keys():

        mu_xt = fixed_params['mu_xt']
        mu = mean_x * (1 - mu_xt) / gain

    else:

        left = baseline - gain / 2
        left = np.where(x > left)[0][0]

        right = baseline + gain / 2
        right = np.where(x < right)[0][-1]

        probability_0_pe = np.sum(y[left:right])
        probability_0_pe /= np.sum(y)
        mu = - np.log(probability_0_pe)

        mu_xt = 1 - gain * mu / mean_x
        mu_xt = max(0.01, mu_xt)

    n_peaks = np.max(x) - (baseline - gain / 2)
    n_peaks = n_peaks / gain
    n_peaks = int(n_peaks)
    amplitude = np.sum(y) * bin_width

    params = {'baseline': baseline, 'sigma_e': sigma_e,
              'sigma_s': sigma_s, 'gain': gain, 'amplitude': amplitude,
              'mu': mu, 'mu_xt': mu_xt, 'n_peaks': n_peaks,
              'bin_width': bin_width}

    if debug:

        x_fit = np.linspace(np.min(x), np.max(x), num=len(x)*10)

        plt.figure()
        plt.step(x, y, where='mid', color='k', label='data')
        plt.errorbar(x, y, y_err, linestyle='None', color='k')
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
    n_pixels = len(pixel_ids)
    n_ac_levels = len(ac_levels)

    timing_histo_filename = 'timing_histo.pk'
    timing_histo_filename = os.path.join(output_path, timing_histo_filename)
    timing_histo = Histogram1D.load(timing_histo_filename)

    results_filename = 'mpe_fit_results'
    results_filename = os.path.join(output_path, results_filename)

    charge_histo_filename = 'charge_histo_ac_level.pk'
    amplitude_histo_filename = 'amplitude_histo_ac_level.pk'
    amplitude_histo_filename = os.path.join(output_path,
                                            amplitude_histo_filename)
    charge_histo_filename = os.path.join(output_path,
                                         charge_histo_filename)

    fit_results_filename = os.path.join(output_path, 'mpe_results{}.npz')

    if n_pixels > 1:

        fit_results_filename = fit_results_filename.format('')
    else:

        fit_results_filename = fit_results_filename.format(pixel_ids[0])

    if n_ac_levels != len(files):

        raise ValueError('n_ac levels = {} != '
                         'n_files = {}'.format(n_ac_levels, len(files)))

    if args['--compute']:

        amplitude = np.zeros((n_ac_levels, n_pixels))
        charge = np.zeros((n_ac_levels, n_pixels))
        time = np.zeros((n_ac_levels, n_pixels))

        charge_histo = Histogram1D(
            bin_edges=np.arange(- 40 * integral_width,
                                2000, bin_width),
            data_shape=(n_ac_levels, n_pixels, ))

        amplitude_histo = Histogram1D(
            bin_edges=np.arange(- 40,
                                500, bin_width),
            data_shape=(n_ac_levels, n_pixels,))

        if os.path.exists(charge_histo_filename):
            raise IOError(
                'File {} already exists'.format(charge_histo_filename))

        if os.path.exists(amplitude_histo_filename):
            raise IOError(
                'File {} already exists'.format(amplitude_histo_filename))

        for i, (file, ac_level) in tqdm(enumerate(zip(files, ac_levels)),
                                        total=n_ac_levels, desc='DAC level',
                                        leave=False):

            time[i] = timing_histo.mode()[pixel_ids]
            pulse_indices = time[i] // 4

            events = calibration_event_stream(file, pixel_id=pixel_ids,
                                              max_events=max_events,
                                              baseline_new=True)
            # events = compute_baseline_with_min(events)
            events = fill_digicam_baseline(events)
            events = subtract_baseline(events)
            # events = find_pulse_with_max(events)
            events = fill_pulse_indices(events, pulse_indices)
            events = compute_charge(events, integral_width, shift)
            events = compute_amplitude(events)

            for event in events:
                charge_histo.fill(event.data.reconstructed_charge,
                                  indices=(i, ))
                amplitude_histo.fill(event.data.reconstructed_amplitude,
                                     indices=(i, ))

        charge_histo.save(charge_histo_filename)
        amplitude_histo.save(amplitude_histo_filename)

        np.savez(fit_results_filename,
                 amplitude=amplitude, charge=charge, time=time,
                 pixel_ids=pixel_ids, ac_levels=ac_levels)

    if args['--fit']:

        input_parameters = yaml.load(open(args['--params'], 'r'))

        gain = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_e = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_s = np.zeros((n_ac_levels, n_pixels)) * np.nan
        baseline = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu_xt = np.zeros((n_ac_levels, n_pixels)) * np.nan
        amplitude = np.zeros((n_ac_levels, n_pixels)) * np.nan

        gain_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_e_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        sigma_s_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        baseline_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        mu_xt_error = np.zeros((n_ac_levels, n_pixels)) * np.nan
        amplitude_error = np.zeros((n_ac_levels, n_pixels)) * np.nan

        chi_2 = np.zeros((n_ac_levels, n_pixels)) * np.nan
        ndf = np.zeros((n_ac_levels, n_pixels)) * np.nan

        ac_limit = [np.inf]*n_pixels

        charge_histo = Histogram1D.load(charge_histo_filename)
        amplitude_histo = Histogram1D.load(amplitude_histo_filename)

        xx = charge_histo.bin_centers
        bin_width = charge_histo.bins[1] - charge_histo.bins[0]

        for i, ac_level in tqdm(enumerate(ac_levels), total=n_ac_levels,
                                desc='DAC level', leave=False):

            for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                    desc='Pixel',
                                    leave=False):

                y = charge_histo.data[i, j]
                y_err = charge_histo.errors()[i, j]

                mask = (y > 0)
                x = xx[mask]
                y = y[mask]
                y_err = y_err[mask]

                if np.max(x) == np.max(xx):

                    # This is to stop the fitting if the values of histogram
                    # reach the last bin

                    continue

                if len(x) == 0:

                    continue

                fit_params_names = describe(mpe_distribution_general)
                options = {'fix_bin_width': True, 'fix_n_peaks': True}
                fixed_params = {'bin_width': bin_width}

                for param in fit_params_names:

                    if param in input_parameters.keys():

                        name = 'fix_' + param

                        options[name] = True
                        fixed_params[param] = input_parameters[param][pixel_id]

                if i > 0:

                    if mu[i - 1, j] > 5:
                        ac_limit[j] = min(i, ac_limit[j])
                        ac_limit[j] = int(ac_limit[j])

                        weights_fit = chi_2[:ac_limit[j], j]
                        weights_fit = weights_fit / ndf[:ac_limit[j], j]

                        options['fix_mu_xt'] = True

                        temp = mu_xt[:ac_limit[j], j] * weights_fit
                        temp = np.nansum(temp)
                        temp = temp / np.nansum(weights_fit)
                        fixed_params['mu_xt'] = temp

                try:

                    chi2 = Chi2Regression(mpe_distribution_general,
                                          x=x, y=y, error=y_err)

                    init_params = compute_init_mpe(x, y, y_err,
                                                   fixed_params=fixed_params,
                                                   debug=debug)
                    limit_params = compute_limit_mpe(init_params)

                    m = Minuit(chi2, **init_params, **limit_params, **options,
                               print_level=0, pedantic=False)

                    m.migrad(ncall=1000)
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
                    amplitude[i, j] = m.values['amplitude']

                    gain_error[i, j] = m.errors['gain']
                    sigma_e_error[i, j] = m.errors['sigma_e']
                    sigma_s_error[i, j] = m.errors['sigma_s']
                    baseline_error[i, j] = m.errors['baseline']
                    mu_error[i, j] = m.errors['mu']
                    mu_xt_error[i, j] = m.errors['mu_xt']
                    amplitude_error[i, j] = m.errors['amplitude']

                    chi_2[i, j] = m.fval
                    ndf[i, j] = len(x) - len(m.list_of_vary_param())

                except Exception as e:

                    print(e)
                    print('Could not fit pixel {} for DAC level'.format(
                        pixel_id, ac_level))

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

        charge_histo = Histogram1D.load(charge_histo_filename)
        charge_histo.draw(index=(0, 0), log=False, legend=False)

        amplitude_histo = Histogram1D.load(amplitude_histo_filename)
        amplitude_histo.draw(index=(0, 0), log=False, legend=False)
        plt.show()

        pass

    return


if __name__ == '__main__':

    entry()
