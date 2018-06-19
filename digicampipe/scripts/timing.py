#!/usr/bin/env python
'''
Do a raw data histogram

Usage:
  timing.py [options] [OUTPUT] [INPUT ...]

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
  -i INPUT --input=INPUT.     Input files.
  -c --compute                Compute the data.
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
from digicampipe.calib.camera.time import compute_time_from_max
from digicampipe.visualization.plot import plot_array_camera, plot_parameter


def compute(files, max_events, pixel_id, output_path, n_samples,
            filename='timing_histo.pk', save=True):

    filename = os.path.join(output_path, filename)

    if os.path.exists(filename) and save:
        raise IOError('The file {} already exists \n'.
                      format(filename))

    n_pixels = len(pixel_id)
    events = calibration_event_stream(files, pixel_id=pixel_id,
                                      max_events=max_events)

    events = compute_time_from_max(events)

    timing_histo = Histogram1D(
        data_shape=(n_pixels, ),
        bin_edges=np.arange(0, n_samples * 4, 4),
        axis_name='reconstructed time [ns]'
    )

    for i, event in enumerate(events):

        timing_histo.fill(event.data.reconstructed_time)

    if save:

        timing_histo.save(filename)

    return timing_histo


def entry():

    args = docopt(__doc__)
    files = args['INPUT']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    pixel_id = convert_pixel_args(args['--pixel'])
    n_samples = int(args['--n_samples'])
    output_path = args['OUTPUT']
    timing_histo_filename = 'timing_histo.pk'

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    if args['--compute']:

        compute(files, max_events, pixel_id, output_path, n_samples,
                timing_histo_filename, save=True)

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

        plt.figure()
        plot_array_camera(pulse_time, label='most probable time of max [ns]',
                          allow_pick=True)

        plt.figure()
        plot_parameter(pulse_time, 'most probable time of max', '[ns]',
                       bins=20)

        plt.show()

    return


if __name__ == '__main__':

    entry()
