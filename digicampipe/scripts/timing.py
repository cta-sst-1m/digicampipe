#!/usr/bin/env python
'''
Do a raw data histogram

Usage:
  digicam-timing [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
  -c --compute                Compute the data.
  -f --fit                    Fit the timing histo.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --save_figures              Save the plots to the OUTPUT folder
  --n_samples=N               Number of samples per waveform
'''
import os
from docopt import docopt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_max_events_args,\
    convert_pixel_args
from digicampipe.visualization.plot import plot_array_camera, plot_parameter
from digicampipe.calib.camera.time import compute_time_from_max, \
    compute_time_from_leading_edge


def compute(files, max_events, pixel_id, n_samples,
            filename='timing_histo.pk', save=True,
            time_method=compute_time_from_max):

    if os.path.exists(filename) and save:
        raise IOError('The file {} already exists \n'.
                      format(filename))

    elif os.path.exists(filename):

        return Histogram1D.load(filename)

    n_pixels = len(pixel_id)
    events = calibration_event_stream(files, pixel_id=pixel_id,
                                      max_events=max_events)

    events = time_method(events)

    timing_histo = Histogram1D(
        data_shape=(n_pixels, ),
        bin_edges=np.arange(0, n_samples * 4, 1),
    )

    for i, event in enumerate(events):

        timing_histo.fill(event.data.reconstructed_time)

    if save:

        timing_histo.save(filename)

    return timing_histo


def entry():

    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    pixel_id = convert_pixel_args(args['--pixel'])
    n_samples = int(args['--n_samples'])
    output_path = args['--output']
    timing_histo_filename = 'timing_histo.pk'
    timing_histo_filename = os.path.join(output_path, timing_histo_filename)
    results_filename = os.path.join(output_path, 'timing_results.npz')

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    if args['--compute']:

        compute(files, max_events, pixel_id, n_samples,
                timing_histo_filename, save=True,
                time_method=compute_time_from_max) # compute_time_from_leading_edge)

    if args['--fit']:

        timing_histo = Histogram1D.load(timing_histo_filename)

        timing = timing_histo.mode()
        timing = timing // 4

        timing[timing <= 4] = np.mean(timing).astype(int)
        timing = timing * 4

        np.savez(results_filename, time=timing)

    if args['--save_figures']:

        histo_path = os.path.join(output_path, timing_histo_filename)
        raw_histo = Histogram1D.load(histo_path)

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

        path = os.path.join(output_path, timing_histo_filename)
        timing_histo = Histogram1D.load(path)
        timing_histo.draw(index=(0, ), log=True, legend=False)

        pulse_time = timing_histo.mode()

        plot_array_camera(pulse_time, label='most probable time [ns]',
                          allow_pick=True)

        plot_parameter(pulse_time, 'most probable time', '[ns]',
                       bins=20)

        pulse_time = np.load(results_filename)['time']

        plot_array_camera(pulse_time, label='time of pulse [ns]',
                          allow_pick=True)

        plot_parameter(pulse_time, 'time of pulse', '[ns]',
                       bins=20)

        plt.show()

    return


if __name__ == '__main__':

    entry()
