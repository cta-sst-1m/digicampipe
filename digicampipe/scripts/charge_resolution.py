#!/usr/bin/env python
"""
Do the charge resolution anaylsis

Usage:
  digicam-charge-resolution [options] [--] <INPUT>...

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
  --charge_linearity=FILE     Charge linearity file
  --save_figures              Save the plots to the OUTPUT folder
  --timing=FILE               Timing npz.
  --saturation_threshold=N    Saturation threshold in LSB
                              [default: 3000]
  --pulse_tail                Use pulse tail for charge integration
"""

import numpy as np
from digicampipe.instrument.light_source import ACLED
from tqdm import tqdm
import os
from docopt import docopt


from digicampipe.utils.docopt import convert_pixel_args, convert_max_events_args, convert_dac_level
from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline, fill_dark_baseline, compute_baseline_shift, _crosstalk_drop_from_baseline_shift, _pde_drop_from_baseline_shift, _gain_drop_from_baseline_shift, compute_nsb_rate
from digicampipe.calib.charge import \
    compute_charge_with_saturation_and_threshold, compute_number_of_pe_from_table, rescale_pulse
from digicampipe.io.event_stream import calibration_event_stream


def charge_to_pe(x, measured_average_charge, true_pe):

    X = measured_average_charge.T
    Y = true_pe.T

    dX = np.diff(X, axis=-1)
    dY = np.diff(Y, axis=-1)

    sign = np.sign(x)

    w = np.clip((np.abs(x[:, None]) - X[:, :-1]) / dX[:, :], 0, 1)

    y = Y[:, 0] + np.nansum(w * dY[:, :], axis=1)
    y = y * sign

    return y


def compute(files, ac_levels, dc_levels, output_filename, dark_charge, dark_baseline,
            max_events, pixels, integral_width, timing, saturation_threshold, pulse_tail, debug):

    directory = '/sst1m/analyzed/calib/mpe/'
    file_calib = os.path.join(directory, 'mpe_fit_results_combined.npz')
    data_calib = dict(np.load(file_calib))

    pe = data_calib['mu']
    pe_err = data_calib['mu_error']
    ac = data_calib['ac_levels'][:, 0]
    ac_led = ACLED(ac, pe, pe_err)
    pde = 0.9  # window filter
    true_pe = ac_led(ac_levels).T * pde

    # mask = true_pe < 5
    # true_pe[mask] = pe[mask]

    n_pixels = len(pixels)
    n_ac_level = len(ac_levels)
    n_dc_level = len(dc_levels)
    n_files = len(files)

    assert n_files == (n_ac_level * n_dc_level)

    shape = (n_dc_level, n_ac_level, n_pixels)
    nsb_mean = np.zeros(shape)
    nsb_std = np.zeros(shape)
    pe_mean = np.zeros(shape)
    pe_std = np.zeros(shape)
    pe_interpolator = lambda x: charge_to_pe(x, dark_charge, true_pe)

    for i, dc_level, in tqdm(enumerate(dc_levels), total=n_dc_level):

        for j, ac_level in tqdm(enumerate(ac_levels), total=n_ac_level):

            index_file = i * n_ac_level + j
            file = files[index_file]
            events = calibration_event_stream(file, max_events=max_events)
            events = fill_dark_baseline(events, dark_baseline[j])
            events = fill_digicam_baseline(events)
            events = compute_baseline_shift(events)
            events = subtract_baseline(events)
            # events = compute_nsb_rate(events, gain, pulse_area, crosstalk,
            #                           bias_resistance, cell_capacitance)
            # events = compute_charge_with_saturation(events, integral_width=7)
            events = compute_charge_with_saturation_and_threshold(events,
                                                                  integral_width=integral_width,
                                                                  debug=debug,
                                                                  trigger_bin=timing,
                                                                  saturation_threshold=saturation_threshold,
                                                                  pulse_tail=pulse_tail)

            events = compute_number_of_pe_from_table(events, pe_interpolator, debug=debug)
            events = rescale_pulse(events, gain_func=_gain_drop_from_baseline_shift,
                                   xt_func=_crosstalk_drop_from_baseline_shift,
                                   pde_func=_pde_drop_from_baseline_shift)
            # events = compute_maximal_charge(events)

            for n, event in enumerate(events):

                pe_mean[i, j] += event.data.reconstructed_number_of_pe
                pe_std[i, j] += event.data.reconstructed_number_of_pe**2
                # nsb_mean[i] += event.data.nsb_rate
                # nsb_std[i] += event.data.nsb_rate**2
                # print(event.data.baseline_shift)

            pe_mean[i, j] = pe_mean[i, j] / (n + 1)
            # nsb_mean[i] = nsb_mean[i] / (n + 1)
            pe_std[i, j] = pe_std[i, j] / (n + 1)
            pe_std[i, j] = np.sqrt(pe_std[i, j] - pe_mean[i, j]**2)
            # nsb_std[i] = nsb_std[i] / (n + 1)
            # nsb_std[i] = np.sqrt(nsb_std[i] - nsb_mean[i]**2)

    np.savez(output_filename, pe_reco_mean=pe_mean, pe_reco_std=pe_std,
             ac_levels=ac_levels, pe=pe, pe_err=pe_err, true_pe=true_pe,
             nsb_mean=nsb_mean, nsb_std=nsb_std)


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
    charge_linearity = args['--charge_linearity']
    charge_linearity = np.load(charge_linearity)

    index_sim = []

    ac_levels_2 = charge_linearity['ac_levels']

    for i, ac in enumerate(ac_levels):

        ind = np.where(ac == ac_levels_2)[0][0]
        index_sim.append(ind)

    dark_charge = charge_linearity['charge_mean'][0][index_sim]
    dark_baseline = charge_linearity['baseline_mean'][0][index_sim]

    if args['--compute']:

        compute(files=files, ac_levels=ac_levels, dc_levels=dc_levels,
                output_filename=output_filename, dark_charge=dark_charge,
                dark_baseline=dark_baseline, max_events=max_events,
                pixels=pixels, integral_width=integral_width, timing=timing,
                saturation_threshold=saturation_threshold,
                pulse_tail=pulse_tail, debug=debug)


if __name__ == '__main__':

    pass

