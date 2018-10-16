#!/usr/bin/env python
"""
Do a raw data histogram

Usage:
  digicam-raw [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o FILE --output=FILE.      File where to store the results.
                              [Default: ./raw_histo.pk]
  -c --compute                Compute the raw data histograms.
  -d --display                Display.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --save_figures              Save the plots to the same folder as output file.
  --baseline_filename=FILE    Output path for DigiCam calculated baseline
                              histogram. If None the histogram will not be
                              computed. FILE should end with '.pk'
                              [Default: None]
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.histogram import Histogram1D
from tqdm import tqdm

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.visualization.plot import plot_histo, plot_array_camera


def compute(files, max_events, pixel_id, filename):
    if os.path.exists(filename) and len(files) == 0:
        raw_histo = Histogram1D.load(filename)
        return raw_histo
    else:
        n_pixels = len(pixel_id)
        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)
        raw_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(0, 4095, 1),
        )

        for event in events:
            raw_histo.fill(event.data.adc_samples)
        raw_histo.save(filename)
        return raw_histo


def compute_baseline_histogram(files, max_events, pixel_id, filename):
    if os.path.exists(filename) and len(files) == 0:
        baseline_histo = Histogram1D.load(filename)
        return baseline_histo
    else:
        n_pixels = len(pixel_id)
        events = calibration_event_stream(files, pixel_id=pixel_id,
                                          max_events=max_events)
        baseline_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(0, 4096, 1 / 16),
        )

        for event in events:
            baseline_histo.fill(event.data.digicam_baseline.reshape(-1, 1))
        baseline_histo.save(filename)

        return baseline_histo


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']

    max_events = convert_max_events_args(args['--max_events'])
    pixel_id = convert_pixel_args(args['--pixel'])
    raw_histo_filename = args['--output']

    baseline_filename = args['--baseline_filename']
    if baseline_filename == 'None':
        baseline_filename = None

    output_path = os.path.dirname(raw_histo_filename)
    if not os.path.exists(output_path):
        raise IOError('Path {} for output '
                      'does not exists \n'.format(output_path))

    if args['--compute']:
        compute(files, max_events, pixel_id, raw_histo_filename)

        if baseline_filename:
            compute_baseline_histogram(files, max_events, pixel_id,
                                       baseline_filename)

    if args['--save_figures']:
        raw_histo = Histogram1D.load(raw_histo_filename)

        path = os.path.join(output_path, 'figures/', 'raw_histo/')

        if not os.path.exists(path):
            os.makedirs(path)

        figure = plt.figure()

        for i, pixel in tqdm(enumerate(pixel_id), total=len(pixel_id)):
            axis = figure.add_subplot(111)
            figure_path = os.path.join(path, 'pixel_{}.pdf')

            try:

                raw_histo.draw(index=(i,), axis=axis, log=True, legend=False)
                figure.savefig(figure_path.format(pixel))

            except Exception as e:

                print('Could not save pixel {} to : {} \n'.
                      format(pixel, figure_path))
                print(e)

            axis.remove()

    if args['--display']:

        raw_histo = Histogram1D.load(raw_histo_filename)

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

    return


if __name__ == '__main__':
    entry()
