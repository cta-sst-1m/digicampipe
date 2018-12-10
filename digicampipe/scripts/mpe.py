#!/usr/bin/env python
"""
Do the Multiple Photoelectron anaylsis and calibrate the AC LEDs

Usage:
  digicam-mpe [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse.
  --fit_output=OUTPUT         File where to store the fit results.
                              [default: ./fit_results.npz]
  --ac_led_filename=OUTPUT    File to store the ACLED calibration
                              [default: ./ac_led.fits]
  --compute_output=OUTPUT     File where to store the compute results.
                              [default: ./charge_histo_ac_level.pk]
  -c --compute                Compute the data.
  -f --fit                    Fit.
  --fit_combine               Fit the histograms in a combined way.
  --ncall=N                   ncall for fit [default: 10000].
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --ac_levels=<DAC>           LED AC DAC level
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --figure_path=FILE          Name of the PDF file to save the figures
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
  --adc_min=N                 Lowest LSB value for the histogram
                              [default: -10]
  --adc_max=N                 Highest LSB value for the histogram
                              [default: 2000]
  --gain=FILE                 Calibration params to use in the fit
  --timing=TIMING_HISTO       Timing histogram
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.fit import HistogramFitter
from histogram.histogram import Histogram1D
from iminuit import describe, Minuit
from tqdm import tqdm
from astropy.table import Table
import fitsio
from scipy.ndimage.filters import convolve1d, convolve
from matplotlib.backends.backend_pdf import PdfPages


from digicampipe.calib.baseline import fill_digicam_baseline, \
    subtract_baseline
from digicampipe.calib.charge import compute_charge, compute_amplitude
from digicampipe.calib.peak import fill_pulse_indices
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, \
    convert_pixel_args, convert_list_int
from digicampipe.utils.pdf import mpe_distribution_general, gaussian, \
    generalized_poisson
from digicampipe.instrument.light_source import ACLED


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

        return mpe_fit(x, baseline, gain, sigma_e, sigma_s, mu, mu_xt, amplitude,
            n_peaks)


class MPECombinedFitter(HistogramFitter):

    def __init__(self, histogram, n_peaks, **kwargs):

        super(MPECombinedFitter, self).__init__(histogram, **kwargs)

        self.n_peaks = n_peaks
        self.histo_mean = histogram.mean()
        self.amplitude = histogram.data.sum(axis=-1)
        # self.mask = mask

        self.parameters_plot_name = {'mu_xt': '$\mu_{XT}$',
                                     'gain': '$G$', 'baseline': '$B$',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$'
                                     }

    def compute_data_bounds(self):

        return self.histogram.bin_centers, self.histogram.data, np.diff(self.histogram.bins)

    def initialize_fit(self):

        pass

    def compute_fit_boundaries(self):

        pass

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, mu_xt):

        mu = (self.histo_mean - baseline) / gain
        mu *= (1 - mu_xt)

        y = mpe_fit(x, baseline=baseline, gain=gain, sigma_e=sigma_e,
                    sigma_s=sigma_s, mu=mu, mu_xt=mu_xt, amplitude=1,
                    n_peaks=self.n_peaks) * self.amplitude[:, np.newaxis]

        return y


def mpe_fit(x, baseline, gain, sigma_e, sigma_s, mu, mu_xt, amplitude, n_peaks):

    if n_peaks > 0:

        x = x - baseline
        photoelectron_peak = np.arange(n_peaks, dtype=np.int)
        sigma_n = sigma_e ** 2 + photoelectron_peak * sigma_s ** 2
        sigma_n = sigma_n
        sigma_n = np.sqrt(sigma_n)

        pdf = generalized_poisson(photoelectron_peak, mu, mu_xt)

        noise = gaussian(x, photoelectron_peak * gain, sigma_n, amplitude=1)

        # print(pdf.shape, b.shape)
        pdf = pdf.dot(noise.T)
        # print(pdf.shape)
        # pdf = np.sum(pdf, axis=1)

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
    ac_led_filename = args['--ac_led_filename']

    timing_filename = args['--timing']

    with fitsio.FITS(timing_filename, 'r') as f:

        timing = f[1]['timing'].read()

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

        for i, ac_level in tqdm(enumerate(ac_levels), total=n_ac_levels,
                                desc='DAC level', leave=False):

            for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                    desc='Pixel',
                                    leave=False):

                histo = Histogram1D.load(charge_histo_filename, row=(i, j))

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
        ac_led = ACLED(ac_levels, mu.T, mu_error.T)
        ac_led.save(ac_led_filename)

    if args['--fit_combine']:

        input_parameters = Table.read(fmpe_results_filename, format='fits')
        input_parameters = input_parameters.to_pandas()

        shape = (n_pixels, n_ac_levels)
        mu = np.zeros(shape) * np.nan
        mu_error = np.zeros(shape) * np.nan
        mean = np.zeros(shape) * np.nan
        std = np.zeros(shape) * np.nan
        chi_2 = np.zeros(shape) * np.nan
        ndf = np.zeros(shape) * np.nan

        gain = np.zeros(n_pixels) * np.nan
        sigma_e = np.zeros(n_pixels) * np.nan
        sigma_s = np.zeros(n_pixels) * np.nan
        baseline = np.zeros(n_pixels) * np.nan
        mu_xt = np.zeros(n_pixels) * np.nan

        gain_error = np.zeros(n_pixels) * np.nan
        sigma_e_error = np.zeros(n_pixels) * np.nan
        sigma_s_error = np.zeros(n_pixels) * np.nan
        baseline_error = np.zeros(n_pixels) * np.nan
        mu_xt_error = np.zeros(n_pixels) * np.nan

        fit_params_names = ['baseline', 'gain', 'sigma_e', 'sigma_s',
                            'mu_xt',
                            ]
        # n_peaks = 300

        saturation_threshold = 300

        for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                desc='Pixel', leave=False):

            histo = Histogram1D.load(charge_histo_filename, rows=(None, j))
            params = {'baseline': input_parameters['baseline'][j],
                      'error_baseline': input_parameters['baseline_error'][j],
                      'limit_baseline': (0, input_parameters['baseline'][j] * 2),
                      'gain': input_parameters['gain'][j],
                      'limit_gain': (input_parameters['gain'][j] * 0.9,
                                     input_parameters['gain'][j] * 1.1),
                      # 'fix_gain': True,
                      'error_gain': input_parameters['gain_error'][j],
                      'sigma_e': input_parameters['sigma_e'][j],
                      'error_sigma_e': input_parameters['sigma_e_error'][j],
                      'limit_sigma_e': (input_parameters['sigma_e'][j] * 0.9,
                                        input_parameters['sigma_e'][j] * 1.1),
                      'sigma_s': input_parameters['sigma_s'][j],
                      'error_sigma_s': input_parameters['sigma_s_error'][j],
                      'limit_sigma_s': (input_parameters['sigma_s'][j] * 0.9,
                                        input_parameters['sigma_s'][j] * 1.1),
                      'mu_xt': 0.1,
                      'limit_mu_xt': (0.001, 0.99),
                      'error_mu_xt': 0.1,
                      }

            mu_max = (histo.max() - params['baseline'])
            mu_max = mu_max / params['gain']

            mask = np.ones(histo.shape[0], dtype=bool)
            mask *= (histo.underflow == 0) * (histo.overflow == 0)
            mask *= (histo.data.sum(axis=-1) > 0)
            mask *= (mu_max <= saturation_threshold)

            mu_max[mu_max > saturation_threshold] = 0
            mu_max = np.nanmax(mu_max)
            n_peaks = int(mu_max) + 10

            print(mask)

            n_histo = int(mask.sum())
            histos = Histogram1D(bin_edges=histo.bins,
                                 data_shape=(n_histo,))
            histos.data = histo.data[mask]

            fitter = MPECombinedFitter(histos, n_peaks=n_peaks,
                                       throw_nan=True, **params)

            fitter.fit()
            print(fitter.parameters, fitter.errors)
            print(fitter.fit_test())

            for i in range(n_ac_levels[mask]):

                fitter.draw_fit(index=i)
            plt.show()

            mu_max = (histo.max() - params['baseline']) / params['gain']

            # mu_init *= (1 - mu_xt)
            mask = np.ones(n_ac_levels, dtype=bool)
            mask *= (histo.underflow == 0) * (histo.overflow == 0)
            mask *= (histo.data.sum(axis=-1) > 0)
            mask *= (mu_max <= saturation_threshold)

            mu_max[mu_max > saturation_threshold] = 0
            mu_max = np.nanmax(mu_max)
            n_peaks = int(mu_max) + 10
            print(n_peaks)
            # mask *= np.isfinite(mu_init)

            for key, val in params.items():

                if not 'limit' in key:

                    mask *= np.isfinite(val)

            # mask[50:] = False
            print(params)

            ndf[j] = np.sum(mask) * len(histo.bin_centers) - len(fit_params_names)

            def cost(param):

                baseline = param[0]
                gain = param[1]
                sigma_e = param[2]
                sigma_s = param[3]
                mu_xt = param[4]
                # mu = histo.param[5:]
                mu = (histo.mean() - baseline) / gain
                mu *= (1 - mu_xt)
                mu = mu[mask]

                x = histo.bin_centers
                count = histo.data
                y_fit = mpe_fit(x, baseline=baseline, gain=gain,
                                sigma_e=sigma_e, sigma_s=sigma_s,
                                mu=mu, mu_xt=mu_xt, amplitude=1,
                                n_peaks=n_peaks)

                # noise = histo[0].data / histo[0].data.sum()
                # origin = histo[0].min()
                # start = np.where(histo.bins == origin)[0][0]
                # end = np.where(histo.bins == histo[0].max())[0][0]
                # print(origin)
                # dac = 30
                # plt.figure()
                # plt.plot(x, y_fit[dac], label='pdf')
                 #print(y_fit.shape)
                # y_fit = convolve1d(y_fit, noise, axis=0, origin=origin)

                # y_fit = convolve1d(y_fit, np.flip(noise), axis=1, origin=origin)
                # print(y_fit[dac])
                # plt.plot(x, count[dac] / np.sum(count[dac]), label='data')
                # plt.plot(x - start, y_fit[dac], label='conv.')
                # plt.plot(x, noise, label='dark')
                # plt.yscale('log')
                # plt.legend()
                y_fit = y_fit * histo.data[mask].sum()
                # plt.show()
                # print(y_fit.shape)
                count = count[mask]
                # cost = count - y_fit

                ## MLE
                mask_empty = count > 0
                cost = 2 * (y_fit - count * np.log(y_fit))
                cost[~mask_empty] = 2 * y_fit[~mask_empty]

                #err = histo.errors()[mask]
                # err[err <= 0] = 1
                # cost *= histo.data[mask] > 0
                # cost = (cost / err)**2
                cost = cost.sum()

                return cost

            m = Minuit(cost, **params,
                       use_array_call=True,
                       forced_parameters=fit_params_names,
                       throw_nan=True)
            m.migrad(ncall=10000, resume=False)
            # print(m.values)
            # print(m)
            # print(m.fval / ndf)

            mean[j, :] = histo.mean()
            std[j, :] = histo.std()
            baseline[j] = m.fitarg['baseline']
            baseline_error[j] = m.fitarg['error_baseline']
            gain[j] = m.fitarg['gain']
            gain_error[j] = m.fitarg['error_gain']
            mu_xt[j] = m.fitarg['mu_xt']
            mu_xt_error[j] = m.fitarg['error_mu_xt']
            sigma_e[j] = m.fitarg['sigma_e']
            sigma_e_error[j] = m.fitarg['error_sigma_e']
            sigma_s[j] = m.fitarg['sigma_s']
            sigma_s_error[j] = m.fitarg['error_sigma_s']

            mu[j, :] = (mean[j] - baseline[j]) / gain[j] * (1 - mu_xt[j])

            del m

        with fitsio.FITS(results_filename, 'rw', clobber=True) as f:

            print(f)
            data = [baseline, gain, mu, mu_xt, sigma_e, sigma_s, std, mean,
                    pixel_ids, baseline_error, gain_error, mu_error,
                    mu_xt_error, sigma_e_error, sigma_s_error, chi_2,
                    ndf]

            names = ['baseline', 'gain', 'mu', 'mu_xt', 'sigma_e', 'sigma_s',
                     'std', 'mean', 'pixel_ids', 'baseline_error',
                     'gain_error', 'mu_error', 'mu_xt_error', 'sigma_e_error',
                     'sigma_s_error', 'chi2', 'ndf']
            f.write(data, names=names, extname='MPE_COMBINED')
            print(f)
            print(f[-1])
            # 0 / 0

    if args['--figure_path'] is not None:

        print('hello')

        pdf = PdfPages(args['--figure_path'])

        with fitsio.FITS(results_filename, 'r') as f:

            for table in f:

                if table.get_extname() == 'MPE_COMBINED':

                    data = {}

                    for key in table.get_colnames():

                        data[key] = table[key].read()

                    for j in range(n_pixels):
                        for i in tqdm(range(n_ac_levels)):

                            histo = Histogram1D.load(charge_histo_filename,
                                                     rows=(i, j))
                            # MPEFitter(histogram=histo)

                            fig = plt.figure()
                            axes = fig.add_subplot(111)
                            histo.draw(axis=axes, log=True, legend=False,
                                       color='k', label='AC level {}\n Pixel {}'
                                                        ''.format(
                                    ac_levels[i], pixel_ids[j]),
                                       x_label='[LSB]')

                            valid = np.isfinite(data['mu'][j, i])

                            if valid:

                                x = histo.bin_centers
                                x_fit = np.linspace(x.min(), x.max(), num=len(x) * 10)
                                y = mpe_fit(x_fit,
                                             baseline=data['baseline'][j],
                                             sigma_e=data['sigma_e'][j],
                                             sigma_s=data['sigma_s'][j],
                                             mu_xt=data['mu_xt'][j],
                                             amplitude=1,
                                             gain=data['gain'][j],
                                             n_peaks=400,
                                             mu=data['mu'][j, i])
                                y = y * histo.data.sum()
                                fit_label = '$\mu$' + ' {0:.3f} [p.e.]\n'.\
                                    format(data['mu'][j, i])
                                fit_label += '$G$' + ' {0:.3f} [LSB / p.e.]\n'.format(
                                    data['gain'][j])
                                fit_label += '$\mu_{xt}$' + ' {0:.3f} [p.e.]\n' \
                                                            ''.format(data['mu_xt'][j])
                                fit_label += '$\sigma_e$' + ' {0:.3f} [LSB]\n' \
                                                            ''.format(data['sigma_e'][j])
                                fit_label += '$\sigma_s$' + ' {0:.3f} [LSB]\n' \
                                                            ''.format(data['sigma_s'][j])
                                fit_label += '$B$' + ' {0:.3f} [LSB]' \
                                                     ''.format(data['baseline'][j])
                                axes.plot(x_fit, y, color='r', label=fit_label)
                                axes.set_ylim(1, histo.data.max() + 10)
                                axes.set_xlim(histo.min() - 1, histo.max() + 1)
                                axes.legend(loc='best')

                            fig.savefig(pdf, format='pdf')
                            plt.close(fig)
        pdf.close()

        fig = plt.figure()
        axes = fig.add_subplot(111)

        ac_led = ACLED.load(ac_led_filename)

        for i in tqdm(range(len(ac_led.y))):

            ac_led.plot(axes=axes, pixel=i)

            figure_name = 'ac_led_pixel_{}'.format(i)
            figure_name = os.path.join(dir_output, figure_name)

            fig.savefig(figure_name)
            fig.clf()

    if args['--display']:

        pixel = 0

        charge_histo = Histogram1D.load(charge_histo_filename)
        charge_histo.draw(index=(0, pixel), log=False, legend=False)

        ac_led = ACLED.load(ac_led_filename)
        ac_led.plot(pixel=pixel)

        plt.show()

    return


if __name__ == '__main__':
    entry()
