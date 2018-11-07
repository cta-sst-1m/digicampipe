#!/usr/bin/env python
"""
Do the Multiple Photoelectron anaylsis

Usage:
  digicam-mpe [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse.
  --fit_output=OUTPUT         File where to store the fit results.
                              [default: ./fit_results.npz]
  --compute_output=OUTPUT     File where to store the compute results.
                              [default: ./charge_histo_ac_level.pk]
  -c --compute                Compute the data.
  -f --fit                    Fit.
  --ncall=N                   ncall for fit [default: 10000].
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
  --adc_min=N                 Lowest LSB value for the histogram
                              [default: -10]
  --adc_max=N                 Highest LSB value for the histogram
                              [default: 2000]
  --gain=<GAIN_RESULTS>       Calibration params to use in the fit
  --timing=<TIMING_HISTO>     Timing histogram
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.fit import HistogramFitter
from histogram.histogram import Histogram1D
from iminuit import describe
from tqdm import tqdm
from astropy.table import Table

from digicampipe.calib.baseline import fill_digicam_baseline, \
    subtract_baseline
from digicampipe.calib.charge import compute_charge, compute_amplitude
from digicampipe.calib.peak import fill_pulse_indices
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, \
    convert_pixel_args, convert_list_int
from digicampipe.utils.pdf import mpe_distribution_general, gaussian, \
    generalized_poisson


class MPEFitter(HistogramFitter):
    def __init__(self, histogram, fixed_params, **kwargs):

        super(MPEFitter, self).__init__(histogram, **kwargs)
        self.initial_parameters = fixed_params
        self.iminuit_options = {**self.iminuit_options, **fixed_params}
        self.parameters_plot_name = {'mu': '$\mu$', 'mu_xt': '$\mu_{XT}$',
                                     'n_peaks': '$N_{peaks}$', 'gain': '$G$',
                                     'amplitude': '$A$', 'baseline': '$B$',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$'
                                     }

    def initialize_fit(self):

        fixed_params = self.initial_parameters
        x = self.bin_centers
        y = self.count

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
        n_peaks = np.round(n_peaks)
        amplitude = np.sum(y)

        params = {'baseline': baseline, 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'gain': gain, 'amplitude': amplitude,
                  'mu': mu, 'mu_xt': mu_xt, 'n_peaks': n_peaks}

        self.initial_parameters = params

    def compute_fit_boundaries(self):

        limit_params = {}

        init_params = self.initial_parameters

        baseline = init_params['baseline']
        gain = init_params['gain']
        sigma_e = init_params['sigma_e']
        sigma_s = init_params['sigma_s']
        mu = init_params['mu']
        amplitude = init_params['amplitude']
        n_peaks = init_params['n_peaks']

        limit_params['limit_baseline'] = (
            baseline - sigma_e, baseline + sigma_e)
        limit_params['limit_gain'] = (0.5 * gain, 1.5 * gain)
        limit_params['limit_sigma_e'] = (0.5 * sigma_e, 1.5 * sigma_e)
        limit_params['limit_sigma_s'] = (0.5 * sigma_s, 1.5 * sigma_s)
        limit_params['limit_mu'] = (0.5 * mu, 1.5 * mu)
        limit_params['limit_mu_xt'] = (0, 0.5)
        limit_params['limit_amplitude'] = (0.5 * amplitude, 1.5 * amplitude)
        limit_params['limit_n_peaks'] = (max(1., n_peaks - 1.), n_peaks + 1.)

        self.boundary_parameter = limit_params

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, mu, mu_xt, amplitude,
            n_peaks):

        if n_peaks > 0:

            x = x - baseline
            photoelectron_peak = np.arange(n_peaks, dtype=np.int)
            sigma_n = sigma_e ** 2 + photoelectron_peak * sigma_s ** 2
            sigma_n = sigma_n
            sigma_n = np.sqrt(sigma_n)

            pdf = generalized_poisson(photoelectron_peak, mu, mu_xt)

            pdf = pdf * gaussian(x, photoelectron_peak * gain, sigma_n,
                                 amplitude=1)
            pdf = np.sum(pdf, axis=-1)

            return pdf * amplitude

        else:

            return np.zeros(x.shape)


def plot_event(events, pixel_id):
    for event in events:
        event.data.plot(pixel_id=pixel_id)
        plt.show()

        yield event


def compute(files, pixel_id, max_events, pulse_indices, integral_width,
            shift, bin_width, charge_histo_filename='charge_histo.pk',
            amplitude_histo_filename='amplitude_histo.pk',
            save=True):
    if os.path.exists(charge_histo_filename) and save:

        raise IOError('File {} already exists'.format(charge_histo_filename))

    elif os.path.exists(charge_histo_filename):

        charge_histo = Histogram1D.load(charge_histo_filename)

    if os.path.exists(amplitude_histo_filename) and save:

        raise IOError(
            'File {} already exists'.format(amplitude_histo_filename))

    elif os.path.exists(amplitude_histo_filename):

        amplitude_histo = Histogram1D.load(amplitude_histo_filename)

    if (not os.path.exists(amplitude_histo_filename)) or \
            (not os.path.exists(charge_histo_filename)):

        n_pixels = len(pixel_id)

        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)
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
                                bin_width))

        amplitude_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(-40, 4096, 1))

        for event in events:

            charge_histo.fill(event.data.reconstructed_charge)
            amplitude_histo.fill(event.data.reconstructed_amplitude)

        if save:
            charge_histo.save(charge_histo_filename)
            amplitude_histo.save(amplitude_histo_filename)

    return amplitude_histo, charge_histo


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_int(args['--max_events'])
    results_filename = args['--fit_output']
    dir_output = os.path.dirname(results_filename)

    if not os.path.exists(dir_output):
        raise IOError('Path {} for output '
                      'does not exists \n'.format(dir_output))

    pixel_ids = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    ncall = int(args['--ncall'])
    ac_levels = convert_list_int(args['--ac_levels'])
    n_pixels = len(pixel_ids)
    n_ac_levels = len(ac_levels)
    adc_min = int(args['--adc_min'])
    adc_max = int(args['--adc_max'])

    timing_filename = args['--timing']
    timing = np.load(timing_filename)['time']

    charge_histo_filename = args['--compute_output']
    fmpe_results_filename = args['--gain']

    if args['--compute']:

        if n_ac_levels != len(files):
            raise ValueError('n_ac_levels = {} != '
                             'n_files = {}'.format(n_ac_levels, len(files)))

        time = np.zeros((n_ac_levels, n_pixels))

        charge_histo = Histogram1D(
            bin_edges=np.arange(adc_min * integral_width,
                                adc_max * integral_width, bin_width),
            data_shape=(n_ac_levels, n_pixels,))

        if os.path.exists(charge_histo_filename):
            raise IOError(
                'File {} already exists'.format(charge_histo_filename))

        for i, (file, ac_level) in tqdm(enumerate(zip(files, ac_levels)),
                                        total=n_ac_levels, desc='DAC level',
                                        leave=False):

            time[i] = timing[pixel_ids]
            pulse_indices = time[i] // 4

            events = calibration_event_stream(file, pixel_id=pixel_ids,
                                              max_events=max_events)
            # events = compute_baseline_with_min(events)
            events = fill_digicam_baseline(events)
            events = subtract_baseline(events)
            # events = find_pulse_with_max(events)
            events = fill_pulse_indices(events, pulse_indices)
            events = compute_charge(events, integral_width, shift)
            events = compute_amplitude(events)

            for event in events:
                charge_histo.fill(event.data.reconstructed_charge,
                                  indices=i)

        charge_histo.save(charge_histo_filename, )

    if args['--fit']:

        input_parameters = Table.read(fmpe_results_filename, format='fits')
        input_parameters = input_parameters.to_pandas()

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

        mean = np.zeros((n_ac_levels, n_pixels)) * np.nan
        std = np.zeros((n_ac_levels, n_pixels)) * np.nan

        chi_2 = np.zeros((n_ac_levels, n_pixels)) * np.nan
        ndf = np.zeros((n_ac_levels, n_pixels)) * np.nan

        ac_limit = [np.inf] * n_pixels

        charge_histo = Histogram1D.load(charge_histo_filename)

        for i, ac_level in tqdm(enumerate(ac_levels), total=n_ac_levels,
                                desc='DAC level', leave=False):

            for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                    desc='Pixel',
                                    leave=False):

                histo = charge_histo[i, pixel_id]

                mean[i, j] = histo.mean()
                std[i, j] = histo.std()

                if histo.overflow > 0 or histo.data.sum() == 0:
                    continue

                fit_params_names = describe(mpe_distribution_general)
                options = {'fix_n_peaks': True}
                fixed_params = {}

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

                    fitter = MPEFitter(histogram=histo, cost='MLE',
                                       pedantic=0, print_level=0,
                                       throw_nan=True,
                                       fixed_params=fixed_params,
                                       **options)

                    fitter.fit(ncall=ncall)

                    if debug:
                        x_label = '[LSB]'
                        label = 'Pixel {}'.format(pixel_id)
                        fitter.draw(legend=False, x_label=x_label, label=label)
                        fitter.draw_init(legend=False, x_label=x_label,
                                         label=label)
                        fitter.draw_fit(legend=False, x_label=x_label,
                                        label=label)
                        plt.show()

                    param = fitter.parameters
                    param_err = fitter.errors
                    gain[i, j] = param['gain']
                    sigma_e[i, j] = param['sigma_e']
                    sigma_s[i, j] = param['sigma_s']
                    baseline[i, j] = param['baseline']
                    mu[i, j] = param['mu']
                    mu_xt[i, j] = param['mu_xt']
                    amplitude[i, j] = param['amplitude']

                    gain_error[i, j] = param_err['gain']
                    sigma_e_error[i, j] = param_err['sigma_e']
                    sigma_s_error[i, j] = param_err['sigma_s']
                    baseline_error[i, j] = param_err['baseline']
                    mu_error[i, j] = param_err['mu']
                    mu_xt_error[i, j] = param_err['mu_xt']
                    amplitude_error[i, j] = param_err['amplitude']

                    chi_2[i, j] = fitter.fit_test() * fitter.ndf
                    ndf[i, j] = fitter.ndf

                except Exception as e:

                    print(e)
                    print('Could not fit pixel {} for DAC level {}'.format(
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
                 ac_levels=ac_levels,
                 amplitude=amplitude,
                 amplitude_error=amplitude_error,
                 mean=mean,
                 std=std,
                 )

    if args['--save_figures']:

        pass

    if args['--display']:

        charge_histo = Histogram1D.load(charge_histo_filename)
        charge_histo.draw(index=(0, 0), log=False, legend=False)

        pass

    return


if __name__ == '__main__':
    entry()
