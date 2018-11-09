#!/usr/bin/env python
"""
Create an histogram of the normalise pulse shape for each pixel.
Usage:
  digicam-pulse-shape [options] <input_files>...

Options:
  -h --help                 Show this screen.
  <input_files>             Path to zfits input files.
  --delays_ns=LIST          Delays used [in ns] for each input file. If set to
                            "none", the timing of each pulse will be
                            determined at the half height of the rising
                            edge. If different of none, the coma separated list
                            must have as many values as inputs files.
                            [default: none]
  --output_hist=PATH        Output histogram file, if not given, we replace the
                            extension of the 1st input file to 'fits.gz'.
  --time_range_ns=LIST      Minimum and maximum time in ns w.r.t. half maximum
                            of the pulse during rise time [default: -10,40].
  --amplitude_range=LIST    Minimum and maximum amplitude of the template
                            normalised in integrated charge [default: -.1,0.4].
  --integration_range=LIST  Minimum and maximum indexes of samples used in the
                            integration for normalization of the pulse charge.
                            If set to "none", all samples are used.
                            [default: none].
  --charge_range=LIST       Minimum and maximum integrated charge in LSB used
                            to build the histogram [default: 1000,8000].
  --n_bin=INT               Number of bins for the 2d histograms
                            [default: 100].
  --disable_bar             use if you want to disable the progress bar.
"""
import numpy as np
from docopt import docopt
import os

from digicampipe.calib.time import estimate_time_from_leading_edge
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.hist2d import Histogram2dChunked
from digicampipe.calib.baseline import fill_digicam_baseline, \
    subtract_baseline, correct_wrong_baseline
from digicampipe.utils.docopt import convert_list_float, convert_list_int, \
    convert_int


def main(
        input_files,
        output_hist,
        delays_ns=None,
        time_range_ns=(-10., 40.),
        amplitude_range=(-0.1, 0.4),
        integration_range=None,
        # charge < 50 pe (noisy) or > 500 pe (saturation) => bad_charge
        # 1 pe <=> 20 LSB integral
        charge_range=(1000., 8000.),
        n_bin=100,
        disable_bar=False
):
    if delays_ns is not None:
        assert len(delays_ns) == len(input_files)
    charge_min = np.min(charge_range)
    charge_max = np.max(charge_range)
    if integration_range is not None:
        integration_min = np.min(integration_range)
        integration_max = np.max(integration_range)
    histo = None
    n_sample = 0
    n_pixel = 0
    for file_idx, input_file in enumerate(input_files):
        if not os.path.isfile(input_file):
            continue
        events = calibration_event_stream([input_file],
                                          disable_bar=disable_bar)
        events = fill_digicam_baseline(events)
        if "SST1M_01_201805" in input_files[0]:  # fix data in May
            print("WARNING: correction of the baselines applied.")
            events = correct_wrong_baseline(events)
        events = subtract_baseline(events)
        for e in events:
            adc = e.data.adc_samples
            if integration_range is not None:
                adc_interp = adc[:, slice(integration_min, integration_max)]
            else:
                adc_interp = adc
            integral = adc_interp.sum(axis=1)
            adc_norm = adc / integral[:, None]
            if delays_ns is None:
                arrival_time_in_ns = estimate_time_from_leading_edge(adc) * 4
            else:
                arrival_time_in_ns = delays_ns[file_idx] * np.ones(1296)
            if histo is None:
                n_pixel, n_sample = adc_norm.shape
                histo = Histogram2dChunked(
                    shape=(n_pixel, n_bin, n_bin),
                    range=[time_range_ns, amplitude_range]
                )
            else:
                assert adc_norm.shape[0] == n_pixel
                assert adc_norm.shape[1] == n_sample
            time_in_ns = np.arange(n_sample) * 4
            bad_charge = np.logical_or(
                integral < charge_min,
                integral > charge_max
            )
            arrival_time_in_ns[bad_charge] = -np.inf  # ignored by histo
            histo.fill(
                x=time_in_ns[None, :] - arrival_time_in_ns[:, None],
                y=adc_norm
            )
    if os.path.exists(output_hist):
        os.remove(output_hist)
    histo.save(output_hist)
    print('2D histogram of pulse shape for all pixel saved as', output_hist)


def entry():
    args = docopt(__doc__)
    inputs = args['<input_files>']
    output_hist = args['--output_hist']
    delays_ns = convert_list_float(args['--delays_ns'])
    time_range_ns = convert_list_float(args['--time_range_ns'])
    amplitude_range = convert_list_float(args['--amplitude_range'])
    integration_range = convert_list_int(args['--integration_range'])
    charge_range = convert_list_float(args['--charge_range'])
    n_bin = convert_int(args['--n_bin'])
    disable_bar = args['--disable_bar']

    if output_hist is None:
        output_hist = os.path.splitext(inputs[0])[0] + '.fits.gz'
    print('options selected:')
    print('input_files:', inputs)
    print('output_hist:', output_hist)
    print('time_range_ns:', time_range_ns)
    print('amplitude_range:', amplitude_range)
    print('integration_range:', integration_range)
    print('charge_range:', charge_range)
    print('n_bin:', n_bin)
    main(
        input_files=inputs,
        output_hist=output_hist,
        delays_ns=delays_ns,
        time_range_ns=time_range_ns,
        amplitude_range=amplitude_range,
        integration_range=integration_range,
        charge_range=charge_range,
        n_bin=n_bin,
        disable_bar=disable_bar
    )


if __name__ == '__main__':
    entry()
