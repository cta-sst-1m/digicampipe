#!/usr/bin/env python
"""
Do the charge linearity anaylsis

Usage:
  digicam-charge-linearity compute --output=FILE --input=FILE --ac_levels=<DAC> --dc_levels=<DAC> [options] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse.
  --output=FILE        File where to store the fit results.
                              [default: ./fit_results.npz]
  --input=FILE
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --ac_levels=<DAC>           LED AC DAC level.
  --dc_levels=<DAC>           LED DC DAC level.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --save_figures              Save the plots to the OUTPUT folder
  --saturation_threshold=N    Saturation threshold in LSB
                              [default: 3000]
  --pulse_tail                Use pulse tail for charge integration
  --integration_method=STR    Integration method (static or dynamic)
                              [Default: dynamic]

Commands:
  compute                     Compute the charge linearity
"""


import numpy as np
from tqdm import tqdm
import os
from docopt import docopt
import fitsio
import matplotlib.pyplot as plt

from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
from digicampipe.calib.charge import compute_dynamic_charge, compute_charge
from digicampipe.calib.peak import fill_pulse_indices
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_list_int, convert_pixel_args, convert_int


def compute(files, ac_levels, dc_levels, output_filename, max_events, pixels,
            integral_width, shift, timing, saturation_threshold,
            pulse_tail=False,
            debug=False, method='dynamic'):

    n_pixels = len(pixels)
    n_files = len(files)
    n_ac_level = len(ac_levels)
    n_dc_level = len(dc_levels)

    assert n_files == (n_ac_level * n_dc_level)

    for file in files:
        assert os.path.exists(file)

    shape = (n_pixels, len(dc_levels), len(ac_levels))
    baseline_mean = np.zeros(shape)
    baseline_std = np.zeros(shape)
    charge_mean = np.zeros(shape)
    charge_std = np.zeros(shape)
    waveform_std = np.zeros(shape)
    ac = np.zeros(shape)
    dc = np.zeros(shape)

    count = np.zeros(shape)

    '''

    for i, dc_level, in tqdm(enumerate(dc_levels), total=n_dc_level,
                             desc='DC level'):

        for j, ac_level in tqdm(enumerate(ac_levels), total=n_ac_level,
                                desc='AC level'):

            index_file = i * n_ac_level + j
            file = files[index_file]

            events = calibration_event_stream(file, max_events=max_events)
            events = fill_digicam_baseline(events)
            events = subtract_baseline(events)
            events = fill_pulse_indices(events, pulse_indices=timing)

            if method == 'dynamic':

                events = compute_dynamic_charge(
                    events,
                    integral_width=integral_width,
                    debug=debug,
                    saturation_threshold=saturation_threshold,
                    pulse_tail=pulse_tail)

            if method == 'static':

                events = compute_charge(events, integral_width=integral_width,
                                        shift=shift)

            for n, event in enumerate(events):

                charge_mean[:, i, j] += event.data.reconstructed_charge
                charge_std[:, i, j] += event.data.reconstructed_charge**2

                baseline_mean[:, i, j] += event.data.baseline
                baseline_std[:, i, j] += event.data.baseline**2

                waveform_std[:, i, j] += event.data.adc_samples[:, 1]**2

            ac[:, i, j] = ac_level
            dc[:, i, j] = dc_level
            count[:, i, j] = n + 1

    charge_mean /= count
    charge_std /= count
    baseline_mean /= count
    baseline_std /= count
    waveform_std /= count

    

    charge_std = np.sqrt(charge_std - charge_mean**2)
    baseline_std = np.sqrt(baseline_std - baseline_mean**2)
    waveform_std = np.sqrt(waveform_std)


    '''

    data = [charge_mean, charge_std, baseline_mean, baseline_std,
            waveform_std, ac, dc]

    names = ['charge_mean', 'charge_std', 'baseline_mean', 'baseline_std',
             'waveform_std', 'ac_levels', 'dc_levels']

    data = dict(zip(names, data))

    np.savez(output_filename, **data)

    # with fitsio.FITS(output_filename, 'rw') as f:

    #  f.write(data, extname='LINEARITY')


def entry():
    args = docopt(__doc__)

    files = args['<INPUT>']
    ac_levels = convert_list_int(args['--ac_levels'])
    dc_levels = convert_list_int(args['--dc_levels'])
    output_filename = args['--output']
    input_filename = args['--input']
    max_events = convert_int(args['--max_events'])
    pixels = convert_pixel_args(args['--pixel'])
    integral_width = float(args['--integral_width'])
    shift = int(args['--shift'])
    saturation_threshold = float(args['--saturation_threshold'])
    pulse_tail = args['--pulse_tail']
    debug = args['--debug']
    method = args['--integration_method']

    if args['compute']:

        with fitsio.FITS(input_filename, 'r') as f:

            timing = f['TIMING']['timing'].read()
            timing = timing // 4

        compute(files=files, ac_levels=ac_levels, dc_levels=dc_levels,
                output_filename=output_filename, max_events=max_events,
                pixels=pixels, integral_width=integral_width, shift=shift,
                timing=timing,
                saturation_threshold=saturation_threshold,
                pulse_tail=pulse_tail, debug=debug, method=method)

    if args['--display']:

        for file in files:

            with fitsio.FITS(file, 'r') as f:

                table = f['LINEARITY'].read()
                data = table

        charge_mean = data['charge_mean']
        charge_std = data['charge_std']
        dc_levels = data['dc_levels']
        ac_levels = data['ac_levels']

        plt.figure()
        plt.scatter(ac_levels, charge_mean, c=dc_levels, s=0.1)
        plt.yscale('log')
        plt.xlabel('AC DAC level')
        plt.ylabel('Charge [a.u.]')
        plt.colorbar(label='DC DAC level')
        plt.ylim(1, None)

        plt.figure()
        plt.scatter(ac_levels, charge_std / charge_mean, c=dc_levels, s=0.1)
        plt.yscale('log')
        plt.xlabel('AC DAC level')
        plt.ylabel(r'$\frac{\sigma_C}{C}$ []')
        plt.colorbar(label='DC DAC level')
        plt.ylim(0.0001, None)

        plt.figure()
        plt.scatter(ac_levels, data['baseline_mean'], c=dc_levels, s=0.1)
        plt.yscale('log')
        plt.xlabel('AC DAC level')
        plt.ylabel(r'Baseline mean [LSB]')
        plt.colorbar(label='DC DAC level')
        plt.ylim(1, None)

        plt.figure()
        plt.scatter(ac_levels, data['baseline_mean'] - data['baseline_mean'][0]
                    , c=dc_levels, s=0.1)
        plt.yscale('log')
        plt.xlabel('AC DAC level')
        plt.ylabel(r'Baseline shift [LSB]')
        plt.colorbar(label='DC DAC level')
        plt.ylim(1, None)

        plt.figure()
        plt.scatter(dc_levels, data['baseline_mean'] - data['baseline_mean'][0]
                    , c=ac_levels, s=0.1)
        plt.yscale('log')
        plt.xlabel('DC DAC level')
        plt.ylabel(r'Baseline shift [LSB]')
        plt.colorbar(label='AC DAC level')
        plt.ylim(1, None)

        plt.show()

    if args['--save_figures']:

        pass
