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
  --fit_summed                Fit the full MPE spectrum.
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
from digicampipe.utils.exception import PeakNotFound
from digicampipe.utils.pdf import fmpe_pdf_10
from digicampipe.utils.fitter import MPEFitter, FMPEFitter, MPECombinedFitter


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
    estimated_gain = 20

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

    if args['--fit_summed']:

        with fitsio.FITS(results_filename, 'rw') as f:

            for i, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                    desc='Pixel'):

                pixel_id = int(pixel_id)
                histo = Histogram1D.load(charge_histo_filename, rows=(None,
                                                                      pixel_id))
                histo = histo.combine()

                try:

                    fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                        throw_nan=True)
                    fitter.fit(ncall=ncall)

                    fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                        initial_parameters=fitter.parameters,
                                        throw_nan=True)
                    fitter.fit(ncall=ncall)

                    data = fitter.results_to_dict()
                    data['pixel_ids'] = np.array(pixel_id)

                    for key, val in data.items():

                        data[key] = val.reshape(1, -1)

                    if debug:

                        x_label = 'Charge [LSB]'
                        label = 'Pixel {}'.format(pixel_id)

                        fitter.draw(x_label=x_label, label=label,
                                    legend=False)
                        fitter.draw_fit(x_label=x_label, label=label,
                                        legend=False)
                        fitter.draw_init(x_label=x_label, label=label,
                                         legend=False)

                        print(data)

                        plt.show()

                    else:

                        if i == 0:

                            f.write(data, extname='FMPE')

                        else:

                            f['FMPE'].append(data)

                except Exception as exception:

                    print('Could not fit FMPE in pixel {}'.format(pixel_id))
                    print(exception)

    if args['--fit']:

        init_param = Table.read(fmpe_results_filename, format='fits',
                                hdu='FMPE')
        init_param = init_param.to_pandas()

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

                    if param in init_param.keys():
                        name = 'fix_' + param

                        options[name] = True
                        fixed_params[param] = init_param[param][pixel_id]

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

        with fitsio.FITS(results_filename, 'rw') as f:

            data = [baseline, gain, mu, mu_xt, sigma_e, sigma_s, std, mean,
                    pixel_ids, baseline_error, gain_error, mu_error,
                    mu_xt_error, sigma_e_error, sigma_s_error, chi_2,
                    ndf]

            names = ['baseline', 'gain', 'mu', 'mu_xt', 'sigma_e', 'sigma_s',
                     'std', 'mean', 'pixel_ids', 'baseline_error',
                     'gain_error', 'mu_error', 'mu_xt_error', 'sigma_e_error',
                     'sigma_s_error', 'chi2', 'ndf']
            f.write(data, names=names, extname='MPE_SINGLE')
            print(f)
            print(f[-1])

        ac_led = ACLED(ac_levels, mu.T, mu_error.T)
        ac_led.save(ac_led_filename)

    if args['--fit_combine']:

        saturation_threshold = 300

        with fitsio.FITS(results_filename, 'rw') as f:

            for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                    desc='Pixel', leave=False):
                pixel_id = int(pixel_id)
                histo = Histogram1D.load(charge_histo_filename, rows=(None,
                                                                      pixel_id))

                names = f['FMPE'].get_colnames()
                init_param = f['FMPE'].read(row=j)[0]
                init_param = dict(zip(names, init_param))

                params = {'baseline': init_param['baseline'],
                          'error_baseline': init_param['error_baseline'],
                          'limit_baseline': (init_param['baseline'] * 0.8,
                                             init_param['baseline'] * 1.2),
                          'gain': init_param['gain'],
                          'limit_gain': (init_param['gain'] * 0.9,
                                         init_param['gain'] * 1.1),
                          'error_gain': init_param['error_gain'],
                          'sigma_e': init_param['sigma_e'],
                          'error_sigma_e': init_param['error_sigma_e'],
                          'limit_sigma_e': (init_param['sigma_e'] * 0.9,
                                            init_param['sigma_e'] * 1.1),
                          'sigma_s': init_param['sigma_s'],
                          'error_sigma_s': init_param['error_sigma_s'],
                          'limit_sigma_s': (init_param['sigma_s'] * 0.9,
                                            init_param['sigma_s'] * 1.1),
                          'mu_xt': 0.1,
                          'limit_mu_xt': (0.001, 0.99),
                          'error_mu_xt': 0.1,
                          }

                if init_param['baseline'] < 0:

                    params['limit_baseline'] = (params['limit_baseline'][1],
                                                params['limit_baseline'][0])

                mu_max = (histo.max() - params['baseline'])
                mu_max = mu_max / params['gain']

                mask = np.ones(histo.shape[0], dtype=bool)
                mask *= (histo.underflow == 0) * (histo.overflow == 0)
                mask *= (histo.data.sum(axis=-1) > 0)
                mask *= (mu_max <= saturation_threshold)

                mu_max[mu_max > saturation_threshold] = 0
                mu_max = np.nanmax(mu_max)
                n_peaks = int(mu_max) + 10

                n_histo = int(mask.sum())
                valid_histo = Histogram1D(bin_edges=histo.bins,
                                          data_shape=(n_histo,))
                valid_histo.data = histo.data[mask]

                fitter = MPECombinedFitter(valid_histo, n_peaks=n_peaks,
                                           throw_nan=True, **params)

                fitter.fit(ncall=10000)

                data = fitter.results_to_dict()
                data['mean'] = histo.mean()
                data['std'] = histo.std()
                data['mu'] = (data['mean'] - data['baseline'])
                data['mu'] /= data['gain']
                data['mu'] *= (1. - data['mu_xt'])
                data['mu'][~mask] = np.nan
                data['error_mu'] = data['mu'] * np.nan
                data['pixel_ids'] = np.array(pixel_id)
                data['n_peaks'] = np.array(n_peaks)
                data['ac_levels'] = np.array(ac_levels)

                for key, val in data.items():

                    data[key] = val.reshape(1, -1)

                if j == 0:

                    f.write(data, extname='MPE_COMBINED')
                else:

                    f['MPE_COMBINED'].append(data)

                print(f)
                print(f['MPE_COMBINED'])
                print(f['MPE_COMBINED']['mu_xt'].read())
                print(f['MPE_COMBINED']['pixel_ids'].read())

    if args['--figure_path'] is not None:

        pdf = PdfPages(args['--figure_path'])

        with fitsio.FITS(results_filename, 'r') as f:

            for table in f:

                if table.get_extname() == 'FMPE':

                    column_names = table.get_colnames()

                    for i, row in enumerate(table):

                        data = dict(zip(column_names, row))
                        fit_params_names = describe(FMPEFitter.pdf)[2:]

                        fit_results = {}
                        fit_errors = {}
                        pixel_id = int(data['pixel_ids'])
                        print(pixel_id)

                        for key in fit_params_names:

                            fit_errors[key] = data['error_' + key]
                            fit_results[key] = data[key]

                        histo = Histogram1D.load(charge_histo_filename,
                                                 rows=(None, pixel_id))
                        histo = histo.combine()

                        fitter = FMPEFitter(histo, estimated_gain=data['gain'])
                        fitter.parameters = fit_results
                        fitter.errors = fit_errors

                        fig = fitter.draw_fit(x_label='[LSB]',
                                              label='FMPE Pixel {}'
                                                    ''.format(pixel_id),
                                              log=True,
                                              legend=False)

                        fig.savefig(pdf, format='pdf')
                        plt.close(fig)

                if table.get_extname() == 'MPE_COMBINED':

                    column_names = table.get_colnames()

                    for i, row in enumerate(table):

                        data = dict(zip(column_names, row))
                        data['error_mu'] = data['mu'] * np.nan
                        fit_params_names = describe(MPECombinedFitter.pdf)[2:]
                        fit_params_names.append('mu')

                        pixel_id = int(data['pixel_ids'])

                        for j in tqdm(range(data['mu'].shape[0])):

                            fit_results = {key: data[key] for key in
                                           fit_params_names}
                            fit_errors = {key: data['error_' + key] for key in
                                          fit_params_names}
                            fit_results['mu'] = fit_results['mu'][j]
                            fit_errors['mu'] = fit_errors['mu'][j]

                            histo = Histogram1D.load(charge_histo_filename,
                                                     rows=(j, pixel_id))

                            valid = np.isfinite(fit_results['mu'])

                            label = 'AC level {}\n Pixel {}' \
                                    ''.format(data['ac_levels'][j],
                                              pixel_id)

                            if valid:

                                fitter = MPECombinedFitter(
                                    histo, n_peaks=data['n_peaks'])
                                fitter.parameters = fit_results
                                fitter.errors = fit_errors

                                fig = fitter.draw_fit(x_label='[LSB]',
                                                      label=label,
                                                      log=True,
                                                      legend=False)

                                fig.savefig(pdf, format='pdf')

                                fig = fitter.draw_fit(x_label='[LSB]',
                                                      label=label,
                                                      log=True,
                                                      legend=False,
                                                      residual=True,
                                                      )
                                fig.savefig(pdf, format='pdf')

                            else:

                                fig = plt.figure()
                                axes = fig.add_subplot(111)
                                histo.draw(axis=axes, log=True, legend=False,
                                           color='k',
                                           label=label,
                                           x_label='[LSB]')

                                fig.savefig(pdf, format='pdf')
                            plt.close(fig)
        pdf.close()

        0/0

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

        pixel_id = 0

        charge_histo = Histogram1D.load(charge_histo_filename)
        charge_histo.draw(index=(0, pixel_id), log=False, legend=False)

        ac_led = ACLED.load(ac_led_filename)
        ac_led.plot(pixel=pixel_id)

        plt.show()

    return


if __name__ == '__main__':
    entry()
