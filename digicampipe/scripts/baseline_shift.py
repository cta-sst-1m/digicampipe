#!/usr/bin/env python
"""
Compute the baseline shift and NSB
Usage:
  digicam-baseline-shift [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  -o OUTPUT --output=OUTPUT   Output file to store the results
                              [default: ./baseline_shift.fits]
  -c --compute                Compute the data.
  -f --fit                    Fit
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --dark=FILE                 Dark histogram
  --dc_levels=<DAC>           LED DC DAC level
  --save_figures              Save the plots to the OUTPUT folder
  --gain=<GAIN_RESULTS>       Calibration params to use in the fit
  --template=<TEMPLATE>       Templates measured
  --crosstalk=<CROSSTALK>     Calibration params to use in the fit
  --integral_width=N          Number of samples used to integrate the pulse
  --sampling_time=N           Sampling period in ns
                              [default: 4]
"""

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from histogram import Histogram1D
from tqdm import tqdm
from fitsio import FITS

from digicampipe.utils.docopt import convert_pixel_args, convert_list_int
from digicampipe.calib.baseline import _compute_nsb_rate
from digicampipe.utils.pulse_template import NormalizedPulseTemplate


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']

    output_filename = args['--output']
    dark_filename = args['--dark']
    pixel_id = convert_pixel_args(args['--pixel'])
    dc_levels = convert_list_int(args['--dc_levels'])
    gain = args['--gain']
    template_filename = args['--template']
    sampling_time = float(args['--sampling_time'])
    integral_width = int(args['--integral_width'])
    crosstalk = args['--crosstalk']
    n_pixels = len(pixel_id)
    n_dc_levels = len(dc_levels)

    dark_histo = Histogram1D.load(dark_filename)

    if args['--compute']:

        if n_dc_levels != len(files):
            raise ValueError('n_dc levels = {} != '
                             'n_files = {}'.format(n_dc_levels, len(files)))

        baseline_mean = np.zeros((n_dc_levels, n_pixels))
        baseline_std = np.zeros((n_dc_levels, n_pixels))

        for i, file in tqdm(enumerate(files), desc='DC level',
                            total=len(files)):

            histo = Histogram1D.load(file)
            baseline_mean[i] = histo.mean()
            baseline_std[i] = histo.std()

        baseline_shift = baseline_mean - dark_histo.mean()

        with FITS(output_filename, 'rw', clobber=True) as f:

            data = [baseline_mean, baseline_std, dc_levels, baseline_shift]

            names = ['baseline_mean', 'baseline_std', 'dc_levels',
                     'baseline_shift']
            f.write(data, names=names)

    if args['--fit']:

        with FITS(output_filename, 'r') as f:

            baseline_shift = f[1].read(columns='baseline_shift')
            baseline_mean = f[1].read(columns='baseline_mean')
            baseline_std = f[1].read(columns='baseline_std')
            dc_levels = f[1].read(columns='dc_levels')

        with FITS(gain, 'r') as f:

            gain = f[1]['gain'].read()

        template = NormalizedPulseTemplate.load(template_filename)
        template_area = template.integral()

        ratio = template.compute_charge_amplitude_ratio(
            integral_width=integral_width, dt_sampling=sampling_time)
        gain = gain * ratio
        crosstalk = 0.08
        bias_resistance = 10 * 1E3
        cell_capacitance = 50 * 1E-15

        nsb_rate = _compute_nsb_rate(baseline_shift=baseline_shift,
                                     gain=gain, pulse_area=template_area,
                                     crosstalk=crosstalk,
                                     bias_resistance=bias_resistance,
                                     cell_capacitance=cell_capacitance)

        with FITS(output_filename, 'rw', clobber=True) as f:

            data = [baseline_mean, baseline_std, dc_levels, baseline_shift,
                    nsb_rate]
            names = ['baseline_mean', 'baseline_std', 'dc_levels',
                     'baseline_shift', 'nsb_rate']
            f.write(data, names=names)

    if args['--display'] or args['--save_figures']:

        with FITS(output_filename, 'r') as f:

            baseline_shift = f[1]['baseline_shift'].read()
            baseline_mean = f[1]['baseline_mean'].read()
            baseline_std = f[1]['baseline_std'].read()
            dc_levels = f[1]['dc_levels'].read()
            nsb_rate = f[1]['nsb_rate'].read()

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

        plt.figure()
        plt.loglog(baseline_shift, nsb_rate)
        plt.xlabel('Baseline shift [LSB]')
        plt.ylabel('$f_{NSB}$ [GHz]')

        if args['--save_figures']:

            pass

        if args['--display']:

            plt.show()

        pass

    return


if __name__ == '__main__':
    entry()
