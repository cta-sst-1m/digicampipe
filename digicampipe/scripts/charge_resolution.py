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

    debug = False
    pulse_tail = False
    shape = (n_dc_level, n_ac_level, n_pixels)
    nsb_mean = np.zeros(shape)
    nsb_std = np.zeros(shape)
    pe_mean = np.zeros(shape)
    pe_std = np.zeros(shape)


    print(dark_baseline, dark_charge)
    pe_interpolator = lambda x: charge_to_pe(x, dark_charge, true_pe)

    for i, dc_level, in tqdm(enumerate(dc_levels), total=n_dc_level):

        for j, ac_level in tqdm(enumerate(ac_levels), total=n_ac_level):

            index_file = i * n_ac_level + j
            file = files[index_file]
            events = calibration_event_stream(file, max_events=max_events)
            events = fill_dark_baseline(events, dark_baseline)
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

            events = compute_number_of_pe_from_table(events, pe_interpolator)
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


if __name__ == '__main__':

    integral_width = 7
    # saturation_threshold = dict(np.load('/home/alispach/Documents/PhD/ctasoft/digicampipe/thresholds.npz'))
    # saturation_threshold = saturation_threshold['threshold_charge']
    # mean = np.nanmean(saturation_threshold)
    # saturation_threshold[np.isnan(saturation_threshold)] = mean

    saturation_threshold = 3000

    max_events = None
    directory = '/sst1m/analyzed/calib/mpe/'
    file_calib = os.path.join(directory, 'mpe_fit_results_combined.npz')
    data_calib = np.load(file_calib)

    # ac_levels = data_calib['ac_levels'][:, 0]
    ac_levels = np.hstack(
        [np.arange(0, 20, 1), np.arange(20, 40, 5), np.arange(45, 450, 5)])
    pde = 0.9  # window filter

    pe = data_calib['mu']
    pe_err = data_calib['mu_error']
    ac_led = ACLED(ac_levels, pe, pe_err)

    ac_levels = np.hstack([np.arange(0, 20, 2), np.arange(20, 450, 10)])

    true_pe = ac_led(ac_levels).T * pde
    # mask = true_pe < 5
    # true_pe[mask] = pe[mask]

    # files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1505, 1557 + 1, 1)]
    files = [
        '/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i)
        for i in range(1982, 2034 + 1, 1)]  # 125 MHz
    # files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(2088, 2140, 1)]  # < 660 MHz

    # files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1350, 1454 + 1, 1)]
    # files = files[100:]
    # ac_levels = ac_levels[100:]
    n_pixels = 1296
    n_files = len(files)

    assert n_files == len(ac_levels)
    filename_1_dark = 'charge_linearity_24102018_dark.npz'
    filename_2 = 'charge_resolution_24102018_125MHz.npz'

    debug = False
    pulse_tail = False
    shape = (n_files, n_pixels)
    nsb_mean = np.zeros(shape)
    nsb_std = np.zeros(shape)
    pe_mean = np.zeros(shape)
    pe_std = np.zeros(shape)

    timing = np.load('/sst1m/analyzed/calib/timing/timing.npz')
    timing = timing['time'] // 4

    data_1 = dict(np.load(filename_1_dark))
    dark_baseline = data_1['baseline_mean'][0]
    charge_mean = data_1['charge_mean']

    print(dark_baseline)
    pe_interpolator = lambda x: charge_to_pe(x, charge_mean, true_pe)

    for i, file in tqdm(enumerate(files), total=n_files):

        events = calibration_event_stream(file, max_events=max_events)
        events = fill_dark_baseline(events, dark_baseline)
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

        events = compute_number_of_pe_from_table(events, pe_interpolator)
        events = rescale_pulse(events, gain_func=_gain_drop_from_baseline_shift,
                               xt_func=_crosstalk_drop_from_baseline_shift,
                               pde_func=_pde_drop_from_baseline_shift)
        # events = compute_maximal_charge(events)

        for n, event in enumerate(events):
            pe_mean[i] += event.data.reconstructed_number_of_pe
            pe_std[i] += event.data.reconstructed_number_of_pe ** 2
            # nsb_mean[i] += event.data.nsb_rate
            # nsb_std[i] += event.data.nsb_rate**2
            # print(event.data.baseline_shift)

        pe_mean[i] = pe_mean[i] / (n + 1)
        # nsb_mean[i] = nsb_mean[i] / (n + 1)
        pe_std[i] = pe_std[i] / (n + 1)
        pe_std[i] = np.sqrt(pe_std[i] - pe_mean[i] ** 2)
        # nsb_std[i] = nsb_std[i] / (n + 1)
        # nsb_std[i] = np.sqrt(nsb_std[i] - nsb_mean[i]**2)

    np.savez(filename_2, pe_reco_mean=pe_mean, pe_reco_std=pe_std,
             ac_levels=ac_levels, pe=pe, pe_err=pe_err, true_pe=true_pe,
             nsb_mean=nsb_mean, nsb_std=nsb_std)

