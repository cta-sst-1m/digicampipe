#!/usr/bin/env python
'''
Do the Multiple Photoelectron anaylsis

Usage:
  digicam-baseline-shift [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o OUTPUT --output=OUTPUT   Folder where to store the results.
  -c --compute                Compute the data.
  -f --fit                    Fit
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --dc_levels=<DAC>           LED DC DAC level
  --save_figures              Save the plots to the OUTPUT folder
  --gain=<GAIN_RESULTS>       Calibration params to use in the fit
  --template=<TEMPLATE>       Templates measured
  --crosstalk=<CROSSTALK>     Calibration params to use in the fit
'''
import os
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args, convert_dac_level


def entry():

    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['--output']

    if not os.path.exists(output_path):

        raise IOError('Path {} for output does not '
                      'exists \n'.format(output_path))

    pixel_id = convert_pixel_args(args['--pixel'])
    dc_levels = convert_dac_level(args['--dc_levels'])
    n_pixels = len(pixel_id)
    n_dc_levels = len(dc_levels)

    results_filename = 'baseline_shift_results.npz'
    results_filename = os.path.join(output_path, results_filename)

    # fmpe_results_filename = args['--gain']
    # crosstalk = args['--crosstalk']
    # templates = args['--template']

    if args['--compute']:

        if n_dc_levels != len(files):
            raise ValueError('n_dc levels = {} != '
                             'n_files = {}'.format(n_dc_levels, len(files)))

        baseline_mean = np.zeros((n_dc_levels, n_pixels))
        baseline_std = np.zeros((n_dc_levels, n_pixels))

        events = calibration_event_stream(files, pixel_id=pixel_id,
                                 max_events=max_events, baseline_new=True)

        for count, event in enumerate(events):

            baseline_mean += event.data.digicam_baseline
            baseline_std += event.data.digicam_baseline**2

        baseline_mean /= count
        baseline_std /= count
        baseline_std -= baseline_mean**2
        baseline_std = np.sqrt(baseline_std)

        np.savez(results_filename, baseline_mean=baseline_mean,
                 baseline_std=baseline_std, dc_levels=dc_levels)

    if args['--fit']:

        pass

    if args['--save_figures']:

        pass

    if args['--display']:

        data = dict(np.load(results_filename))
        baseline_mean = data['baseline_mean']
        baseline_std = data['baseline_std']
        dc_levels = data['dc_levels']

        plt.figure()
        plt.plot(dc_levels, baseline_mean)


        pass

    return


if __name__ == '__main__':

    entry()
