#!/usr/bin/env python
"""
Do a raw data histogram

Usage:
  digicam-raw compute --output=FILE [options] <INPUT>...
  digicam-raw display <INPUT>
  digicam-raw save_figure --output=FILE <INPUT>
  digicam-raw fit --output=FILE <INPUT>

Options:
    -h --help                   Show this screen.
    --max_events=N              Maximum number of events to analyse
    -o FILE --output=FILE.      File where to store the results.
    -p --pixel=<PIXEL>          Give a list of pixel IDs.
    --baseline_subtracted       Perform baseline subtraction to the raw data
    --baseline_filename=FILE    Output path for DigiCam calculated baseline
                                histogram. If "none" the histogram will not be
                                computed. FILE should end with '.pk'
                                [Default: none]
    --event_types=<TYPE>        Comma separated list of integers corresponding
                                to the events types that are taken into the
                                histogram (others are discarded).
                                If set to "none", all events are included.
                                [Default: none]
    --disable_bar               If used, the progress bar is not show while
                                reading files.

Commands:
    compute                     Compute the histogram
    display                     Display the histogram
    save_figure                 Save the figures to the output
    fit                         Fit a gaussian on to the histogram
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.histogram import Histogram1D
from tqdm import tqdm
from fitsio import FITS

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, convert_pixel_args, \
    convert_list_int, convert_text
from digicampipe.visualization.plot import plot_histo, plot_array_camera
from digicampipe.utils.fitter import GaussianFitter


def compute(files, filename, max_events=None, pixel_id=None, event_types=None,
            disable_bar=False, baseline_subtracted=False):
    if os.path.exists(filename) and len(files) == 0:
        raw_histo = Histogram1D.load(filename)
        return raw_histo
    else:
        if pixel_id is None:
            pixel_id = convert_pixel_args(None)
        n_pixels = len(pixel_id)
        events = calibration_event_stream(
            files, pixel_id=pixel_id, max_events=max_events,
            disable_bar=disable_bar)
        if baseline_subtracted:
            bin_edges = np.arange(-100, 4095, 1)
        else:
            bin_edges = np.arange(0, 4095, 1)
        raw_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=bin_edges,
        )

        for event in events:
            if event_types and event.event_type not in event_types:
                continue
            samples = event.data.adc_samples
            if baseline_subtracted:
                samples = samples - event.data.digicam_baseline[:, None]
            raw_histo.fill(samples)
        raw_histo.save(filename)

        return raw_histo


def compute_baseline_histogram(files, filename, max_events=None, pixel_id=None,
                               event_types=None, disable_bar=False):
    if os.path.exists(filename) and len(files) == 0:
        baseline_histo = Histogram1D.load(filename)
        return baseline_histo
    else:
        if pixel_id is None:
            pixel_id = convert_pixel_args(None)
        n_pixels = len(pixel_id)
        events = calibration_event_stream(
            files, pixel_id=pixel_id, max_events=max_events,
            disable_bar=disable_bar
        )
        baseline_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(0, 4096, 1 / 16),
        )

        for event in events:
            if event_types and event.event_type not in event_types:
                continue
            baseline_histo.fill(event.data.digicam_baseline.reshape(-1, 1))
        baseline_histo.save(filename)

        return baseline_histo


def fit_gaussian(filename, output, debug=False):

    histos = Histogram1D.load(filename)
    n_pixels = histos.shape[0]
    colnames = ['mean', 'error_mean', 'sigma', 'error_sigma', 'amplitude',
                'error_amplitude', 'chi_2', 'ndf']
    data = {key: np.zeros(n_pixels) for key in colnames}

    for i in tqdm(range(n_pixels), total=n_pixels, desc='Pixel'):

        histo = histos[i]

        try:

            fitter = GaussianFitter(histo)
            fitter.fit()
            results = fitter.results_to_dict()

            if debug:

                fitter.draw_init()
                fitter.draw_fit()
                plt.show()

        except Exception as e:

            results = {key: np.array(np.nan) for key in colnames}

        for key, val in results.items():

            data[key][i] = val

    print(data)

    with FITS(output, 'rw') as f:

        f.write(data=data, extname='RAW')

    return


def entry():
    args = docopt(__doc__)

    files = args['<INPUT>']
    max_events = convert_int(args['--max_events'])
    pixel_id = convert_pixel_args(args['--pixel'])
    base_sub = args['--baseline_subtracted']
    output = args['--output']
    event_types = convert_list_int(args['--event_types'])
    baseline_filename = convert_text(args['--baseline_filename'])
    disable_bar = args['--disable_bar']

    output_path = os.path.dirname(output)
    if not os.path.exists(output_path) and output_path != "":
        raise IOError('Path {} for output '
                      'does not exists \n'.format(output_path))

    if args['compute']:
        compute(
            files=files,
            filename=output,
            max_events=max_events,
            pixel_id=pixel_id,
            event_types=event_types,
            disable_bar=disable_bar,
            baseline_subtracted=base_sub
        )
        if baseline_filename:
            compute_baseline_histogram(
                files=files,
                filename=baseline_filename,
                max_events=max_events,
                pixel_id=pixel_id,
                disable_bar=disable_bar
            )

    if args['fit']:

        fit_gaussian(files[0], output)

    if args['save_figure']:

        raw_histo = Histogram1D.load(files[0])
        raw_histo.save_figures(output, log=True, x_label='[LSB]')

    if args['display']:

        raw_histo = Histogram1D.load(files[0])
        pixel = 0
        raw_histo.draw(index=(pixel,), log=True, legend=False,
                       label='Histogram {}'.format(pixel), x_label='[LSB]')
        mean_value = raw_histo.mean()
        plot_histo(mean_value, bins='auto', x_label='Mean value [LSB]')
        plot_array_camera(mean_value, label='Mean value [LSB]')
        if baseline_filename:
            baseline_histo = Histogram1D.load(baseline_filename)
            baseline_histo.draw(index=(pixel,), log=True, legend=False,
                                label='Histogram {}'.format(pixel),
                                x_label='DigiCam baseline [LSB]')
            mean_baseline = baseline_histo.mean()
            plot_histo(mean_baseline, bins='auto',
                       x_label='Mean DigiCam baseline [LSB]')
            plot_array_camera(mean_baseline,
                              label='Mean DigiCam baseline [LSB]')
            plot_array_camera(mean_baseline - mean_value,
                              label='Diff [LSB]')
            plot_histo(mean_baseline - mean_value, bins='auto',
                       x_label='Diff [LSB]')
        plt.show()


if __name__ == '__main__':
    entry()
