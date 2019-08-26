#!/usr/bin/env python
"""
Do the Multiple Photoelectron anaylsis and calibrate the AC LEDs

Usage:
  digicam-mpe compute --output=FILE --ac_levels=<DAC> --calib=FILE [--pixel=<PIXEL> --shift=N --integral_width=N --adc_min=N --adc_max=N] <INPUT>...
  digicam-mpe fit combined --output=FILE --ac_levels=<DAC> [--pixel=<PIXEL> --estimated_gain=N --ncall=N] [options] <INPUT>
  digicam-mpe fit single --output=FILE --ac_levels=<DAC> [--pixel=<PIXEL> --estimated_gain=N --ncall=N] [options] <INPUT>
  digicam-mpe display <INPUT>
  digicam-mpe combine --output=FILE <INPUT>...
  digicam-mpe save_figure --output=FILE --ac_levels=<DAC> --calib=FILE <INPUT>

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse.
  -o --output=FILE            Output file.
  --ncall=N                   ncall for fit [default: 10000].
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --ac_levels=<DAC>           LED AC DAC level
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
  --adc_min=N                 Lowest LSB value for the histogram
                              [default: -10]
  --adc_max=N                 Highest LSB value for the histogram
                              [default: 2000]
  --calib=FILE                Calibration FITS filename
  --estimated_gain=N          Estimated value for gain
                              [default: 20]
Commands:
  compute                     Compute the histogram
  fit
  combined
  single
  save_figure
  display
  combine                     Combine the output to a single file
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.histogram import Histogram1D
from iminuit import describe
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
from digicampipe.utils.pdf import mpe_distribution_general
from digicampipe.instrument.light_source import ACLED
from digicampipe.utils.fitter import MPEFitter, FMPEFitter, MPECombinedFitter

N_PIXELS = 1296


def compute_pe(charge, baseline, crosstalk, gain):

    mu = (charge - baseline) / gain * (1. - crosstalk)

    return mu


def compute_pe_error(charge, baseline, crosstalk, gain,
                     charge_error, baseline_error, crosstalk_error,
                     gain_error, mu):

    mu_error = np.abs(1 / (charge - baseline)) * (charge_error + baseline_error)
    mu_error += np.abs(1 / gain) * gain_error
    mu_error += np.abs(1 / (1 - crosstalk)) * crosstalk_error
    mu_error = mu_error * mu

    return mu_error


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


def fit_summed_mpe(histo_filename, pixel_ids, estimated_gain=20, ncall=10000,
                   debug=False,
                   ):

    n_pixels = len(pixel_ids)

    names = ['baseline', 'error_baseline', 'gain',
             'error_gain', 'sigma_e', 'error_sigma_e', 'sigma_s',
             'error_sigma_s', 'a_0', 'error_a_0', 'a_1',
             'error_a_1', 'a_2', 'error_a_2', 'a_3', 'error_a_3',
             'a_4', 'error_a_4', 'a_5', 'error_a_5', 'a_6',
             'error_a_6', 'a_7', 'error_a_7', 'a_8', 'error_a_8',
             'a_9', 'error_a_9', 'chi_2', 'ndf', 'pixel_ids']
    formats = ['f8'] * (len(names) - 2)
    formats = formats + ['i8', 'i8']

    data = {key: np.zeros(N_PIXELS, dtype=formats[i]) * np.nan for i, key in enumerate(names)}
    data['pixel_ids'] = np.arange(N_PIXELS)

    for i, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                            desc='Pixel'):

        pixel_id = int(pixel_id)
        histo = Histogram1D.load(histo_filename, rows=(None, pixel_id))
        histo = histo.combine()

        try:

            fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                throw_nan=True)
            fitter.fit(ncall=ncall)

            fitter = FMPEFitter(histo, estimated_gain=estimated_gain,
                                initial_parameters=fitter.parameters,
                                throw_nan=True)
            fitter.fit(ncall=ncall)

            results = fitter.results_to_dict()
            for key, val in results.items():

                data[key][pixel_id] = val

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

        except Exception as exception:

            print('Could not fit MPE_SUMMED in pixel {}'.format(pixel_id))
            print(exception)

    return data


def fit_combined_mpe(histo_filename, pixel_ids, init_params,
                     ac_levels, saturation_threshold=300, ncall=10000,
                     debug=False):

    n_pixels = len(pixel_ids)
    n_ac_levels = len(ac_levels)

    names = ['baseline', 'error_baseline', 'gain', 'error_gain', 'sigma_e',
             'error_sigma_e', 'sigma_s', 'error_sigma_s', 'mu_xt',
             'error_mu_xt', 'chi_2', 'ndf', 'mean', 'std',
             'n_peaks', 'ac_levels']

    data = {key: np.zeros(N_PIXELS) * np.nan for key in names}
    data['pixel_ids'] = np.arange(N_PIXELS)
    data['mean'] = np.zeros((N_PIXELS, n_ac_levels)) * np.nan
    data['std'] = np.zeros((N_PIXELS, n_ac_levels)) * np.nan
    data['mu'] = np.zeros((N_PIXELS, n_ac_levels)) * np.nan
    data['error_mu'] = np.zeros((N_PIXELS, n_ac_levels)) * np.nan
    data['ac_levels'] = np.zeros((N_PIXELS, n_ac_levels)) * np.nan
    data['ac_levels'][:] = ac_levels

    for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                            desc='Pixel', leave=False):

        try:

            init_param = {}
            for key, val in init_params.items():

                init_param[key] = val[pixel_id]

            pixel_id = int(pixel_id)
            histo = Histogram1D.load(histo_filename, rows=(None, pixel_id))
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

            mu_max = compute_pe(histo.max(),
                                baseline=params['baseline'],
                                crosstalk=0, gain=params['gain'])
            mu_estimated = compute_pe(histo.mean(),
                                      baseline=params['baseline'],
                                      crosstalk=0.08,
                                      gain=params['gain'])

            mask = np.ones(histo.shape[0], dtype=bool)
            mask *= (histo.underflow == 0) * (histo.overflow == 0)
            mask *= (histo.data.sum(axis=-1) > 0)
            mask *= (mu_max <= saturation_threshold)
            mask *= (mu_estimated > 1)

            mu_max[mu_max > saturation_threshold] = 0
            mu_max = np.nanmax(mu_max)
            n_peaks = int(mu_max) + 10

            n_histo = int(mask.sum())
            valid_histo = Histogram1D(bin_edges=histo.bins,
                                      data_shape=(n_histo,))
            valid_histo.data = histo.data[mask]

            fitter = MPECombinedFitter(valid_histo, n_peaks=n_peaks,
                                       throw_nan=True, **params)
            fitter.fit(ncall=ncall)

            results = fitter.results_to_dict()
            for key, val in results.items():

                data[key][pixel_id] = val

            n_entries = histo.data.sum(axis=-1)

            data['mean'][pixel_id] = histo.mean()
            data['std'][pixel_id] = histo.std()
            mean_error = data['std'][pixel_id] / np.sqrt(n_entries)
            data['mu'][pixel_id] = compute_pe(
                data['mean'][pixel_id],
                baseline=data['baseline'][pixel_id],
                gain=data['gain'][pixel_id],
                crosstalk=data['mu_xt'][pixel_id])

            data['mu'][pixel_id][~mask] = np.nan
            data['error_mu'][pixel_id] = compute_pe_error(
                data['mean'][pixel_id],
                baseline=data['baseline'][pixel_id],
                gain=data['gain'][pixel_id],
                crosstalk=data['mu_xt'][pixel_id],
                charge_error=mean_error,
                baseline_error=data['error_baseline'][pixel_id],
                gain_error=data['error_gain'][pixel_id],
                crosstalk_error=data['error_mu_xt'][pixel_id],
                mu=data['mu'][pixel_id])

            data['error_mu'][pixel_id][~mask] = np.nan
            data['n_peaks'][pixel_id] = n_peaks

        except Exception as e:

            print('Could not fit pixel {}'.format(pixel_id))
            raise e

        if debug:

            for i in range(n_ac_levels):

                histo = Histogram1D.load(histo_filename, rows=(i, pixel_id))

                fitter_debug = MPECombinedFitter(
                    histo, n_peaks=data['n_peaks'][pixel_id])
                fitter_debug.parameters = fitter.parameters
                fitter_debug.errors = fitter.errors

                fitter_debug.draw_fit(x_label='[LSB]',
                                  log=False,
                                  legend=False)
                plt.show()

            print(results)
            print(data)

    return data


def fit_single_mpe(histo_filename, ac_levels, pixel_ids, init_params,
                   ncall=10000, debug=False):

    n_pixels = N_PIXELS
    n_ac_levels = len(ac_levels)

    names = ['baseline', 'gain', 'mu', 'mu_xt', 'sigma_e', 'sigma_s',
             'std', 'mean', 'pixel_ids', 'baseline_error',
             'gain_error', 'mu_error', 'mu_xt_error', 'sigma_e_error',
             'sigma_s_error', 'chi2', 'ndf']

    data = {name: np.zeros((n_ac_levels, n_pixels)) * np.nan for name in names}
    ac_limit = [np.inf] * n_pixels

    for i, ac_level in tqdm(enumerate(ac_levels), total=n_ac_levels,
                            desc='DAC level', leave=False):

        for j, pixel_id in tqdm(enumerate(pixel_ids), total=n_pixels,
                                desc='Pixel',
                                leave=False):

            histo = Histogram1D.load(histo_filename, row=(i, j))
            data['mean'][i, j] = histo.mean()
            data['std'][i, j] = histo.std()

            if histo.overflow > 0 or histo.data.sum() == 0:
                continue

            fit_params_names = describe(mpe_distribution_general)
            options = {'fix_n_peaks': True}
            fixed_params = {}

            for param in fit_params_names:

                if param in init_params.keys():
                    name = 'fix_' + param

                    options[name] = True
                    fixed_params[param] = init_params[param][pixel_id]

            if i > 0:

                if data['mu'][i - 1, j] > 5:
                    ac_limit[j] = min(i, ac_limit[j])
                    ac_limit[j] = int(ac_limit[j])

                    weights_fit = data['chi_2'][:ac_limit[j], j]
                    weights_fit = weights_fit / data['ndf'][:ac_limit[j], j]

                    options['fix_mu_xt'] = True

                    temp = data['mu_xt'][:ac_limit[j], j] * weights_fit
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

                param = fitter.results_to_dict()

                for key, val in param.items():

                    data[key][i, j] = val

            except Exception as e:

                print(e)
                print('Could not fit pixel {} for DAC level {}'.format(
                    pixel_id, ac_level))

    return data


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_int(args['--max_events'])
    output = args['--output']

    pixel_ids = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    ncall = int(args['--ncall'])
    ac_levels = convert_list_int(args['--ac_levels'])
    n_pixels = len(pixel_ids)

    if ac_levels is not None:
        n_ac_levels = len(ac_levels)
    adc_min = int(args['--adc_min'])
    adc_max = int(args['--adc_max'])
    calib_filename = args['--calib']
    estimated_gain = float(args['--estimated_gain'])

    if args['compute']:

        with fitsio.FITS(calib_filename, 'r') as f:

            timing = f['TIMING']['timing'].read()

        if n_ac_levels != len(files):
            raise ValueError('n_ac_levels = {} != '
                             'n_files = {}'.format(n_ac_levels, len(files)))

        time = np.zeros((n_ac_levels, n_pixels))

        charge_histo = Histogram1D(
            bin_edges=np.arange(adc_min * integral_width,
                                adc_max * integral_width, bin_width),
            data_shape=(n_ac_levels, n_pixels,))

        if os.path.exists(output):
            raise IOError(
                'File {} already exists'.format(output))

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

        charge_histo.save(output)

    if args['fit']:

        data = {}

        data_summed = fit_summed_mpe(histo_filename=files[0],
                                     pixel_ids=pixel_ids,
                                     estimated_gain=estimated_gain,
                                     ncall=ncall,
                                     debug=debug)

        if args['combined']:

            data_combined = fit_combined_mpe(histo_filename=files[0],
                                             pixel_ids=pixel_ids,
                                             init_params=data_summed,
                                             ac_levels=ac_levels,
                                             ncall=ncall,
                                             debug=debug)
            data['MPE_COMBINED'] = data_combined

        if args['single']:

            data_single = fit_single_mpe(histo_filename=files[0],
                                         ac_levels=ac_levels,
                                         pixel_ids=pixel_ids,
                                         init_params=data_summed,
                                         ncall=ncall,
                                         debug=debug)
            data['MPE_SINGLE'] = data_single

        with fitsio.FITS(output, 'rw') as f:

            if 'MPE_SUMMED' not in f:

                data['MPE_SUMMED'] = data_summed

            for analysis, result in data.items():

                f.write_table(data=result,
                              extname=analysis,
                              table_type='binary')

    if args['combine']:

        output_data = {}

        for i, file in enumerate(files):

            with fitsio.FITS(file, 'r') as f_in:

                for j, table in enumerate(f_in):

                    if j == 0:

                        continue

                    ext = table.get_extname()

                    if i == 0:

                        output_data[ext] = {}

                    for col in table.get_colnames():

                        if i == 0:
                            output_data[ext][col] = table.read(columns=col)

                        output_data[ext][col][i] = table.read(columns=col,
                                                              rows=i)

        with fitsio.FITS(output, 'rw', clobber=True) as f_out:

            for table_name, data in output_data.items():

                f_out.write_table(data, extname=table_name)

    if args['save_figure']:

        pdf = PdfPages(args['--output'])

        with fitsio.FITS(calib_filename, 'r') as f:

            for table in f:

                if table.get_extname() == 'MPE_SUMMED':

                    column_names = table.get_colnames()

                    for i, row in enumerate(table):

                        data = dict(zip(column_names, row))
                        fit_params_names = describe(FMPEFitter.pdf)[2:]

                        fit_results = {}
                        fit_errors = {}
                        pixel_id = int(data['pixel_ids'])
                        valid = True

                        for key in fit_params_names:

                            fit_errors[key] = data['error_' + key]
                            fit_results[key] = data[key]

                            if not np.isfinite(fit_results[key]):

                                valid = False
                        if not valid:

                            continue

                        histo = Histogram1D.load(files[0],
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

                            valid = np.isfinite(fit_results['mu'])

                            if not valid:

                                continue

                            histo = Histogram1D.load(files[0],
                                                     rows=(j, pixel_id))

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

    if args['display']:

        pixel_id = 0

        charge_histo = Histogram1D.load(files[0])
        charge_histo.draw(index=(0, pixel_id), log=False, legend=False)


        plt.show()

    return


if __name__ == '__main__':
    entry()
