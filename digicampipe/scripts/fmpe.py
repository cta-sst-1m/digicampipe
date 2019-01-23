#!/usr/bin/env python
"""
Do Full Multiple Photoelectron anaylsis

Usage:
  digicam-fmpe [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyze
  -c --compute                Compute the data.
  -f --fit                    Fit.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixels=<PIXEL>         Give a list of pixel IDs.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --figure_path=FILE          Path to save the figures
                              [default: None]
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
  --ncall=N                   Number of calls for the fit [default: 10000]
  --n_samples=N               Number of samples in readout window
  --charge_histo_filename=FILE
  --amplitude_histo_filename=FILE
  --results_filename=FILE
  --estimated_gain=N          Estimated gain for the fit
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.histogram import Histogram1D
from tqdm import tqdm
import pandas as pd
from iminuit.util import describe
import fitsio
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages

from digicampipe.scripts.mpe import compute as mpe_compute
from digicampipe.utils.docopt import convert_int, \
    convert_pixel_args, convert_text
from digicampipe.utils.fitter import FMPEFitter

from digicampipe.visualization.plot import plot_array_camera, plot_histo


def compute(files, max_events, pixel_id, n_samples, results_filename,
            charge_histo_filename, amplitude_histo_filename, save,
            integral_width, shift, bin_width):

    with fitsio.FITS(results_filename, 'r') as f:

        pulse_indices = f['TIMING']['timing'].read() // 4

    amplitude_histo, charge_histo = mpe_compute(
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

    _, fig_1 = plot_array_camera(gain, label='Gain [LSB $\cdot$ ns / p.e.]')
    _, fig_2 = plot_array_camera(sigma_e, label='$\sigma_e$ [LSB $\cdot$ ns]')
    _, fig_3 = plot_array_camera(sigma_s, label='$\sigma_s$ [LSB $\cdot$ ns]')
    _, fig_7 = plot_array_camera(baseline, label='Baseline [LSB]')

    fig_4 = plot_histo(gain, x_label='Gain [LSB $\cdot$ ns / p.e.]', bins='auto')
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


def plot_fit(histo, fit_results, label=''):

    fitter = FMPEFitter(histo, throw_nan=True,
                        estimated_gain=fit_results['gain'])

    for key in fitter.parameters_name:

        fitter.parameters[key] = fit_results[key]
        fitter.errors[key] = fit_results[key + '_error']

    fitter.ndf = fit_results['ndf']

    x_label = 'Charge [LSB]'
    fig = fitter.draw_fit(x_label=x_label, label=label, legend=False,
                          log=True)

    return fig


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_int(args['--max_events'])

    pixel_id = convert_pixel_args(args['--pixels'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    figure_path = convert_text(args['--figure_path'])
    n_pixels = len(pixel_id)

    charge_histo_filename = args['--charge_histo_filename']
    amplitude_histo_filename = args['--amplitude_histo_filename']
    results_filename = args['--results_filename']
    n_samples = int(args['--n_samples'])
    ncall = int(args['--ncall'])
    estimated_gain = float(args['--estimated_gain'])

    if args['--compute']:
        compute(files,
                max_events=max_events,
                pixel_id=pixel_id,
                n_samples=n_samples,
                results_filename=results_filename,
                charge_histo_filename=charge_histo_filename,
                amplitude_histo_filename=amplitude_histo_filename,
                save=True,
                integral_width=integral_width,
                shift=shift,
                bin_width=bin_width)

    if args['--fit']:

        param_names = describe(FMPEFitter.pdf)[2:]
        param_error_names = [key + '_error' for key in param_names]
        columns = param_names + param_error_names
        columns = columns + ['chi_2', 'ndf']
        data = np.zeros((n_pixels, len(columns))) * np.nan

        results = pd.DataFrame(data=data, columns=columns)

        for i, pixel in tqdm(enumerate(pixel_id), total=n_pixels,
                             desc='Pixel'):

            histo = Histogram1D.load(charge_histo_filename, rows=i)

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

            with fitsio.FITS(results_filename, 'rw', clobber=True) as f:

                f.write(results.to_records(index=False), extname='FMPE')

    if figure_path:

        figure_dir = os.path.dirname(figure_path)
        plot_results(results_filename, figure_dir)

        pdf = PdfPages(figure_path)
        fit_results = Table.read(results_filename, format='fits')
        fit_results = fit_results.to_pandas()

        for i, pixel in tqdm(enumerate(pixel_id), total=len(pixel_id)):

            fit_result = fit_results.iloc[int(pixel)]
            charge_histo = Histogram1D.load(charge_histo_filename,
                                            rows=int(pixel))

            label = 'Pixel {}'.format(pixel)
            try:

                fig = plot_fit(charge_histo, fit_result, label)
                fig.savefig(pdf, format='pdf')
            except Exception as e:

                print('Could not save pixel {} to : {} \n'.
                      format(pixel, figure_path))
                print(e)
        pdf.close()

    if args['--display']:

        pixel = 0
        charge_histo = Histogram1D.load(charge_histo_filename, rows=int(pixel))
        fit_results = Table.read(results_filename, format='fits')
        fit_results = fit_results.to_pandas()
        fit_result = fit_results.iloc[pixel]

        plot_results(results_filename)
        plot_fit(charge_histo, fit_result)

        plt.show()

        pass

    return


if __name__ == '__main__':
    entry()
