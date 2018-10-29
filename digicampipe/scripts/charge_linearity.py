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
"""


import numpy as np
from tqdm import tqdm
import os
from docopt import docopt

from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
from digicampipe.calib.charge import \
    compute_charge_with_saturation_and_threshold
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_dac_level, convert_pixel_args, convert_max_events_args


def compute(files, ac_levels, dc_levels, output_filename, max_events, pixels,
         integral_width, timing, saturation_threshold, pulse_tail, debug):

    n_pixels = len(pixels)
    n_files = len(files)
    n_ac_level = len(ac_levels)
    n_dc_level = len(dc_levels)

    assert n_files == (n_ac_level * n_dc_level)

    for file in files:
        assert os.path.exists(file)

    assert not os.path.exists(output_filename)

    shape = (len(dc_levels), len(ac_levels), n_pixels)
    amplitude_mean = np.zeros(shape)
    amplitude_std = np.zeros(shape)
    baseline_mean = np.zeros(shape)
    baseline_std = np.zeros(shape)
    charge_mean = np.zeros(shape)
    charge_std = np.zeros(shape)

    waveform_std = np.zeros(shape)

    for i, dc_level, in tqdm(enumerate(dc_levels), total=n_dc_level):

        for j, ac_level in tqdm(enumerate(ac_levels), total=n_ac_level):

            index_file = i * n_ac_level + j
            file = files[index_file]

            events = calibration_event_stream(file, max_events=max_events)
            events = fill_digicam_baseline(events)
            events = subtract_baseline(events)
            # events = compute_charge_with_saturation(events, integral_width=7)

            events = compute_charge_with_saturation_and_threshold(
                events, integral_width=integral_width, debug=debug,
                trigger_bin=timing, saturation_threshold=saturation_threshold,
                pulse_tail=pulse_tail)
            # events = compute_maximal_charge(events)

            for n, event in enumerate(events):

                charge_mean[i, j] += event.data.reconstructed_charge
                amplitude_mean[i, j] += event.data.reconstructed_amplitude

                charge_std[i, j] += event.data.reconstructed_charge**2
                amplitude_std[i, j] += event.data.reconstructed_amplitude**2

                baseline_mean[i, j] += event.data.baseline
                baseline_std[i, j] += event.data.baseline**2
                waveform_std[i, j] += np.std(event.data.adc_samples[:, :6])

            charge_mean[i, j] = charge_mean[i, j] / (n + 1)
            waveform_std[i, j] = waveform_std[i, j] / (n + 1)
            charge_std[i, j] = charge_std[i, j] / (n + 1)
            charge_std[i, j] = np.sqrt(charge_std[i, j] - charge_mean[i, j]**2)
            amplitude_mean[i, j] = amplitude_mean[i, j] / (n + 1)
            amplitude_std[i, j] = amplitude_std[i, j] / (n + 1)
            amplitude_std[i, j] = np.sqrt(amplitude_std[i, j] - amplitude_mean[i, j]**2)
            baseline_mean[i, j] = baseline_mean[i, j] / (n + 1)
            baseline_std[i, j] = baseline_std[i, j] / (n + 1)
            baseline_std[i, j] = np.sqrt(baseline_std[i, j] - baseline_mean[i, j]**2)

    np.savez(output_filename, charge_mean=charge_mean, charge_std=charge_std,
             amplitude_mean=amplitude_mean, amplitude_std=amplitude_std,
             ac_levels=ac_levels, dc_levels=dc_levels,
             baseline_mean=baseline_mean,
             baseline_std=baseline_std, waveform_std=waveform_std)


def entry():
    args = docopt(__doc__)

    files = args['<INPUT>']
    ac_levels = convert_dac_level(args['--ac_levels'])
    dc_levels = convert_dac_level(args['--dc_levels'])
    output_filename = args['--output_file']
    max_events = convert_max_events_args(args['--max_events'])
    pixels = convert_pixel_args(args['--pixel'])
    integral_width = float(args['--integral_width'])
    timing = args['--timing']
    timing = np.load(timing)['time'] // 4
    saturation_threshold = float(args['--saturation_threshold'])
    pulse_tail = args['--pulse_tail']
    debug = args['--debug']

    if args['--compute']:

        compute(files=files, ac_levels=ac_levels, dc_levels=dc_levels,
             output_filename=output_filename, max_events=max_events,
             pixels=pixels, integral_width=integral_width, timing=timing,
             saturation_threshold=saturation_threshold, pulse_tail=pulse_tail,
             debug=debug)


if __name__ == '__main__':
    integral_width = 7
    # saturation_threshold = dict(np.load('/home/alispach/Documents/PhD/ctasoft/digicampipe/thresholds.npz'))
    # saturation_threshold = saturation_threshold['threshold_charge']

    # mean = np.nanmean(saturation_threshold)
    # saturation_threshold[np.isnan(saturation_threshold)] = mean

    saturation_threshold = 3000

    max_events = 50
    pixels = np.arange(1296)
    debug = False
    pulse_tail = False
    output_filename = 'charge_linearity_29102018_all.npz'
    timing = np.load('/sst1m/analyzed/calib/timing/timing.npz')
    timing = timing['time'] // 4
    # files = ['/home/alispach/Downloads/ac_scan/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1350, 1454 + 1, 1)] # 0 MHz no window
    # files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1505, 1557 + 1, 1)] # 0 MHz window
    # files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1982, 2034 + 1, 1)] # 125 MHz
    # files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(2088, 2140, 1)]  # < 660 MHz
    ac_levels = np.hstack([np.arange(0, 20, 2), np.arange(20, 450, 10)])
    dc_levels = np.arange(200, 320, 10)
    files = [
        '/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i)
        for i in range(1505, 2141)]

    compute(files=files, ac_levels=ac_levels, dc_levels=dc_levels,
         output_filename=output_filename, max_events=max_events,
         pixels=pixels, integral_width=integral_width, timing=timing,
         saturation_threshold=saturation_threshold, pulse_tail=pulse_tail,
         debug=debug)