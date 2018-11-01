#!/usr/bin/env python
"""
Do the charge linearity anaylsis

Usage:
  digicam-charge-linearity [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse.
  --output_file=OUTPUT        File where to store the fit results.
                              [default: ./fit_results.npz]
  -c --compute                Compute the data.
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
  --timing=FILE               Timing npz.
  --saturation_threshold=N    Saturation threshold in LSB
                              [default: 3000]
  --pulse_tail                Use pulse tail for charge integration
  --integration_method=STR    Integration method (static or dynamic)
                              [Default: dynamic]
"""


import numpy as np
from tqdm import tqdm
import os
from docopt import docopt
import fitsio

from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
from digicampipe.calib.charge import compute_dynamic_charge, compute_charge
from digicampipe.calib.peak import fill_pulse_indices
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_dac_level, convert_pixel_args, convert_max_events_args


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

    shape = (len(dc_levels), len(ac_levels), n_pixels)
    baseline_mean = np.zeros(shape)
    baseline_std = np.zeros(shape)
    charge_mean = np.zeros(shape)
    charge_std = np.zeros(shape)
    waveform_std = np.zeros(shape)

    count = np.zeros(shape[:-1])

    for i, dc_level, in tqdm(enumerate(dc_levels), total=n_dc_level):

        for j, ac_level in tqdm(enumerate(ac_levels), total=n_ac_level):

            index_file = i * n_ac_level + j
            file = files[index_file]

            events = calibration_event_stream(file, max_events=max_events)
            events = fill_digicam_baseline(events)
            events = subtract_baseline(events)

            if method == 'dynamic':

                events = compute_dynamic_charge(
                    events,
                    integral_width=integral_width,
                    debug=debug,
                    trigger_bin=timing - shift,
                    saturation_threshold=saturation_threshold,
                    pulse_tail=pulse_tail)

            if method == 'static':

                events = fill_pulse_indices(events, pulse_indices=timing)
                events = compute_charge(events, integral_width=integral_width,
                                        shift=shift)

            for n, event in enumerate(events):

                charge_mean[i, j] += event.data.reconstructed_charge
                charge_std[i, j] += event.data.reconstructed_charge**2

                baseline_mean[i, j] += event.data.baseline
                baseline_std[i, j] += event.data.baseline**2

                waveform_std[i, j] += event.data.adc_samples[:, 1]**2

            count[i, j] = n + 1

    charge_mean /= count[..., None]
    charge_std /= count[..., None]
    baseline_mean /= count[..., None]
    baseline_std /= count[..., None]
    waveform_std /= count[..., None]

    charge_std = np.sqrt(charge_std - charge_mean**2)
    baseline_std = np.sqrt(baseline_std - baseline_mean**2)
    waveform_std = np.sqrt(waveform_std)

    with fitsio.FITS(output_filename, 'rw', clobber=True) as f:

        data = [charge_mean, charge_std, baseline_mean, baseline_std,
                waveform_std, ac_levels, dc_levels, count]

        names = ['charge_mean', 'charge_std', 'baseline_mean', 'baseline_std',
                 'waveform_std', 'ac_levels', 'dc_levels', 'count']

        data = dict(zip(names, data))

        for key, val in data.items():

            f.write(val, extname=key)


def entry():
    args = docopt(__doc__)

    files = args['<INPUT>']
    ac_levels = convert_dac_level(args['--ac_levels'])
    dc_levels = convert_dac_level(args['--dc_levels'])
    output_filename = args['--output_file']
    max_events = convert_max_events_args(args['--max_events'])
    pixels = convert_pixel_args(args['--pixel'])
    integral_width = float(args['--integral_width'])
    shift = int(args['--shift'])
    timing = args['--timing']
    timing = np.load(timing)['time'] // 4
    saturation_threshold = float(args['--saturation_threshold'])
    pulse_tail = args['--pulse_tail']
    debug = args['--debug']
    method = args['--integration_method']

    if args['--compute']:

        compute(files=files, ac_levels=ac_levels, dc_levels=dc_levels,
                output_filename=output_filename, max_events=max_events,
                pixels=pixels, integral_width=integral_width, shift=shift,
                timing=timing,
                saturation_threshold=saturation_threshold,
                pulse_tail=pulse_tail, debug=debug, method=method)

    if args['--display']:

        with fitsio.FITS(output_filename, 'r') as f:

            data = {hdu.get_extname(): hdu.read() for hdu in f}

        pass

    if args['--save_figures']:

        pass
