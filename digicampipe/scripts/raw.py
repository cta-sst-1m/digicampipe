#!/usr/bin/env python
'''
Do a raw data histogram

Usage:
  digicam-raw [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
                              [Default: .]
  -c --compute                Compute the data.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --save_figures              Save the plots to the OUTPUT folder
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


def compute(files, max_events, pixel_id, filename):

    if os.path.exists(filename):

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


def entry():

    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    pixel_id = convert_pixel_args(args['--pixel'])
    output_path = args['--output']
    raw_histo_filename = os.path.join(output_path, 'raw_histo.pk')

    if not os.path.exists(output_path):

        raise IOError('Path {} for output '
                      'does not exists \n'.format(output_path))

    if args['--compute']:

        compute(files, max_events, pixel_id, raw_histo_filename)

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
        raw_histo.draw(index=(pixel, ), log=True, legend=False,
                       label='Histogram {}'.format(pixel), x_label='[LSB]')
        plt.show()

    return


if __name__ == '__main__':

    entry()
