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

'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from ctapipe.io import HDF5TableWriter
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.io.containers_calib import SPEResultContainer
from histogram.histogram import Histogram1D
from .spe import compute_gaussian_parameters_first_peak
from digicampipe.calib.camera.baseline import fill_baseline, \
    fill_digicam_baseline, subtract_baseline
from digicampipe.calib.camera.peak import find_pulse_with_max, find_pulse_gaussian_filter, find_pulse_1, find_pulse_2, find_pulse_3, find_pulse_4
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude, fit_template
from digicampipe.utils.docopt import convert_max_events_args, convert_pixel_args


def plot_event(events, pixel_id):

    for event in events:

        event.data.plot(pixel_id=pixel_id)
        plt.show()

        yield event


def entry():

    args = docopt(__doc__)
    files = args['INPUT']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['OUTPUT']

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    pixel_id = convert_pixel_args(args['--pixel'])
    n_pixels = len(pixel_id)

    amplitude_histo_filename = output_path + 'amplitude_histo.pk'
    charge_histo_filename = output_path + 'charge_histo.pk'

    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])

    n_samples = int(args['--n_samples']) # TODO access this in a better way !

    if args['--compute']:

        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)

        # events = compute_baseline_with_min(events)
        events = fill_digicam_baseline(events)
        events = subtract_baseline(events)
        events = find_pulse_with_max(events)
        events = compute_charge(events, integral_width, shift)

        charge_histo = Histogram1D(
                            data_shape=(n_pixels,),
                            bin_edges=np.arange(-4096 * integral_width,
                                                4096 * integral_width),
                            axis_name='[LSB]'
                        )
    if args['--fit']:

        spe_charge = Histogram1D.load(charge_histo_filename)
        spe_amplitude = Histogram1D.load(amplitude_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        dark_count_rate = np.zeros(n_pixels) * np.nan
        electronic_noise = np.zeros(n_pixels) * np.nan

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels):

            x = max_histo._bin_centers()
            y = max_histo.data[i]

            n_entries = np.sum(y)

            mask = (y > 0)
            x = x[mask]
            y = y[mask]

            try:

                val, err = compute_gaussian_parameters_first_peak(x,
                                                                  y,
                                                                  snr=3,
                                                                  )

                number_of_zeros = val['amplitude']
                window_length = 4 * n_samples
                rate = compute_dark_rate(number_of_zeros,
                                         n_entries,
                                         window_length)
                dark_count_rate[pixel] = rate
                electronic_noise[pixel] = val['sigma']

            except Exception as e:

                print('Could not compute dark count rate in pixel {}'
                      .format(pixel))
                print(e)

        np.savez(dark_count_rate_filename, dcr=dark_count_rate)
        np.savez(electronic_noise_filename, electronic_noise)

        spe = spe_charge
        name = 'charge'
        crosstalk = np.zeros(n_pixels) * np.nan

        results = SPEResultContainer()

        table_name = 'analysis_' + name

        with HDF5TableWriter(results_filename, table_name, mode='w') as h5:

            for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels):

                try:

                    x = spe._bin_centers()
                    y = spe.data[i]
                    y_err = spe.errors(index=i)
                    sigma_e = electronic_noise[i]
                    sigma_e = sigma_e if not np.isnan(sigma_e) else None

                    params, params_err, params_init, params_bound = \
                        fit_spe(x, y, y_err, sigma_e=sigma_e,
                                snr=3, debug=debug)

                    for key, val in params.items():

                        setattr(results.init, key, params_init[key])
                        setattr(results.param, key, params[key])
                        setattr(results.param_errors, key, params_err[key])

                    for key, val in results.items():
                        results[key]['pixel_id'] = pixel

                    for key, val in results.items():

                        h5.write('spe_' + key, val)

                    n_entries = params['a_1']
                    n_entries += params['a_2']
                    n_entries += params['a_3']
                    n_entries += params['a_4']
                    crosstalk[pixel] = (n_entries - params['a_1']) / n_entries
                    print(crosstalk[pixel])
                    print(params['sigma_e'])

                except Exception as e:

                    print('Could not fit for pixel_id : {}'.format(pixel))
                    print(e)

            np.savez(crosstalk_filename, crosstalk)

    if args['--save_figures']:

        spe_charge = Histogram1D.load(charge_histo_filename)
        spe_amplitude = Histogram1D.load(amplitude_histo_filename)
        raw_histo = Histogram1D.load(raw_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        figure_directory = output_path + 'figures/'

        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        histograms = [spe_charge, spe_amplitude, raw_histo, max_histo]
        names = ['histogram_charge/', 'histogram_amplitude/', 'histogram_raw/',
                 'histo_max/']

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

        spe_charge = Histogram1D.load(charge_histo_filename)
        spe_amplitude = Histogram1D.load(amplitude_histo_filename)
        raw_histo = Histogram1D.load(raw_histo_filename)
        max_histo = Histogram1D.load(max_histo_filename)

        spe_charge.draw(index=(0, ), log=True, legend=False)
        spe_amplitude.draw(index=(0, ), log=True, legend=False)
        raw_histo.draw(index=(0, ), log=True, legend=False)
        max_histo.draw(index=(0, ), log=True, legend=False)

        try:
            df = pd.HDFStore(results_filename, mode='r')
            parameters = df['analysis_charge/spe_param']
            parameters_error = df['analysis_charge/spe_param_errors']

            dark_count_rate = np.load(dark_count_rate_filename)['dcr']
            crosstalk = np.load(crosstalk_filename)['arr_0']
        except FileNotFoundError as e:

            print(e)
            print('Could not find the analysis files !')
            plt.show()
            exit()

        label = 'mean : {:2f} \n std : {:2f}'

        for key, val in parameters.items():

            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.hist(val, bins='auto', label=label.format(np.mean(val),
                                                           np.std(val)))
            axes.set_xlabel(key + ' []')
            axes.set_ylabel('count []')
            axes.legend(loc='best')

            # fig = plt.figure()
            # axes = fig.add_subplot(111)
            # axes.hist(parameters_error[key])
            # axes.set_xlabel(key + '_error' + ' []')
            # axes.set_ylabel('count []')

        crosstalk = crosstalk[np.isfinite(crosstalk)]
        dark_count_rate = dark_count_rate[np.isfinite(dark_count_rate)]

        plt.figure()
        plt.hist(crosstalk, bins='auto', label=label.format(np.mean(crosstalk),
                                                            np.std(crosstalk)))
        plt.xlabel('XT []')
        plt.legend(loc='best')

        plt.figure()
        plt.hist(dark_count_rate[np.isfinite(dark_count_rate)],
                 bins='auto',
                 label=label.format(np.mean(dark_count_rate),
                                    np.std(dark_count_rate)))
        plt.xlabel('dark count rate [GHz]')
        plt.legend(loc='best')

        plt.show()

    return


if __name__ == '__main__':

    entry()
