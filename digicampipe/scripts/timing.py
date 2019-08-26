#!/usr/bin/env python
"""
Do histogram of the pulse reconstructed time. In each bin is represented the
number of times the reconstructed time within time bin.

Usage:
  digicam-timing [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  --timing_histo_filename=FILE.  Folder where to store the results.
  --ac_levels=<DAC>           LED AC DAC level
  -c --compute                Compute the data.
  -f --fit                    Fit the timing histo.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --save_figures              Save the plots to the OUTPUT folder
  --n_samples=N               Number of samples per waveform
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram.histogram import Histogram1D
from tqdm import tqdm
from scipy.stats import mode

from digicampipe.calib.time import compute_time_from_max
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, \
    convert_pixel_args, convert_list_int
from digicampipe.visualization.plot import plot_array_camera, plot_parameter


def compute(files, max_events, pixel_id, n_samples, ac_levels,
            filename='timing_histo.pk', save=True,
            time_method=compute_time_from_max):
    if os.path.exists(filename) and save:
        raise IOError('The file {} already exists \n'.
                      format(filename))

    elif os.path.exists(filename):

        return Histogram1D.load(filename)

    n_pixels = len(pixel_id)
    n_ac_levels = len(ac_levels)
    n_files = len(files)

    if n_ac_levels != n_files:

        raise IOError('Number of files = {} does not match number of DAC level'
                      ' {}'.format(n_files, n_ac_levels))

    timing_histo = Histogram1D(
        data_shape=(n_ac_levels, n_pixels,),
        bin_edges=np.arange(0, n_samples * 4, 1),
    )

    for i, file in tqdm(enumerate(files), total=n_ac_levels, desc='DAC level'):

        events = calibration_event_stream(file, pixel_id=pixel_id,
                                          max_events=max_events)

        events = time_method(events)

        for event in events:

            timing_histo.fill(event.data.reconstructed_time, indices=i)

    if save:
        timing_histo.save(filename)

    return timing_histo


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']

    max_events = convert_int(args['--max_events'])
    pixel_id = convert_pixel_args(args['--pixel'])
    n_samples = int(args['--n_samples'])
    timing_histo_filename = args['--timing_histo_filename']
    ac_levels = convert_list_int(args['--ac_levels'])

    output_path = os.path.dirname(timing_histo_filename)
    results_filename = os.path.join(output_path, 'timing.npz')

    if not os.path.exists(output_path):
        raise IOError('Path for output does not exists \n')

    if args['--compute']:
        compute(files, max_events, pixel_id, n_samples, ac_levels,
                timing_histo_filename, save=True,
                time_method=compute_time_from_max)
        # or try to use compute_time_from_leading_edge)

    if args['--fit']:
        timing_histo = Histogram1D.load(timing_histo_filename)

        timing = timing_histo.mode()
        timing = mode(timing, axis=0)[0][0]

        np.savez(results_filename, time=timing)

    if args['--save_figures']:

        raw_histo = Histogram1D.load(timing_histo_filename)

        path = os.path.join(output_path, 'figures/', 'timing_histo/')

        if not os.path.exists(path):
            os.makedirs(path)

        figure = plt.figure()

        for i, pixel in tqdm(enumerate(pixel_id), total=len(pixel_id)):
            axis = figure.add_subplot(111)
            figure_path = path + 'pixel_{}.pdf'

            try:

                raw_histo.draw(index=(i,), axis=axis, log=True, legend=False)
                figure.savefig(figure_path.format(pixel))

            except Exception as e:

                print('Could not save pixel {} to : {} \n'.
                      format(pixel, figure_path))
                print(e)

            axis.remove()

    if args['--display']:

        timing_histo = Histogram1D.load(timing_histo_filename)
        timing_histo.draw(index=(len(ac_levels)-1, 0, ), log=True, legend=False)

        pulse_time = timing_histo.mode()

        plt.figure()
        plt.plot(ac_levels, pulse_time)
        plt.xlabel('DAC level')
        plt.ylabel('Reconstructed pulse time [ns]')

        pulse_time = np.load(results_filename)['time']

        plot_array_camera(pulse_time, label='time of pulse [ns]',
                          allow_pick=True)

        plot_parameter(pulse_time, 'time of pulse', '[ns]',
                       bins=20)

        plt.show()

    return


if __name__ == '__main__':
    entry()
