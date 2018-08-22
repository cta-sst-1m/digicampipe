#!/usr/bin/env python
"""
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
"""
import os
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from histogram.histogram import Histogram1D

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

    histo = Histogram1D(data_shape=(n_dc_levels, n_pixels),
                        bin_edges=np.arange(0, 4096))

    if args['--compute']:

        if n_dc_levels != len(files):
            raise ValueError('n_dc levels = {} != '
                             'n_files = {}'.format(n_dc_levels, len(files)))

        baseline_mean = np.zeros((n_dc_levels, n_pixels))
        baseline_std = np.zeros((n_dc_levels, n_pixels))

        for i, file in tqdm(enumerate(files), desc='DC level',
                            total=len(files)):

            events = calibration_event_stream(file, pixel_id=pixel_id,
                                              max_events=max_events,
                                              baseline_new=True)

            for count, event in enumerate(events):

                baseline_mean[i] += event.data.digicam_baseline
                baseline_std[i] += event.data.digicam_baseline**2

                histo.fill(event.data.adc_samples, indices=(i, ))

            count += 1
            baseline_mean[i] = baseline_mean[i] / count
            baseline_std[i] = baseline_std[i] / count
            baseline_std[i] = baseline_std[i] - baseline_mean[i]**2
            baseline_std[i] = np.sqrt(baseline_std[i])

        histo.save(os.path.join(output_path, 'raw_histo.pk'))
        np.savez(results_filename, baseline_mean=baseline_mean,
                 baseline_std=baseline_std, dc_levels=dc_levels)

    if args['--fit']:

        data = dict(np.load(results_filename))
        baseline_mean = data['baseline_mean']
        baseline_std = data['baseline_std']
        dc_levels = data['dc_levels']

        gain = 5
        template_area = 18
        crosstalk = 0.08
        bias_resistance = 10 * 1E3
        cell_capacitance = 50 * 1E-15

        baseline_shift = baseline_mean - baseline_mean[0]
        nsb_rate = baseline_shift / gain / template_area * (1 - crosstalk)

        nsb_rate = gain * template_area / (baseline_shift * (1 - crosstalk))
        nsb_rate = nsb_rate - cell_capacitance * bias_resistance
        nsb_rate = 1 / nsb_rate

        np.savez(results_filename, baseline_mean=baseline_mean,
                 baseline_std=baseline_std, dc_levels=dc_levels,
                 nsb_rate=nsb_rate, baseline_shift=baseline_shift)

    if args['--save_figures']:

        pass

    if args['--display']:

        data = dict(np.load(results_filename))
        histo = Histogram1D.load(os.path.join(output_path, 'raw_histo.pk'))
        baseline_mean = histo.mean()
        baseline_std = histo.std()
        nsb_rate = data['nsb_rate']

        print(baseline_mean.shape)

        histo.draw((0, 1))
        histo.draw((49, 1))

        plt.figure()
        plt.plot(dc_levels, baseline_mean)
        plt.xlabel('DC DAC level')
        plt.ylabel('Baseline mean [LSB]')

        plt.figure()
        plt.plot(dc_levels, baseline_std)
        plt.xlabel('DC DAC level')
        plt.ylabel('Baseline std [LSB]')

        plt.figure()
        plt.plot(nsb_rate, baseline_std)
        plt.xlabel('$f_{NSB}$ [GHz]')
        plt.ylabel('Baseline std [LSB]')

        plt.figure()
        plt.plot(baseline_mean, baseline_std)
        plt.xlabel('Baseline mean [LSB]')
        plt.ylabel('Baseline std [LSB]')

        plt.figure()
        plt.semilogy(dc_levels, nsb_rate)
        plt.xlabel('DC DAC level')
        plt.ylabel('$f_{NSB}$ [GHz]')
        plt.show()

        pass

    return


if __name__ == '__main__':

    entry()
