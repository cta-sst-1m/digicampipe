#!/usr/bin/env python
'''
Reconstruct the pulse template

Usage:
  digicam-template [options] <input_files>...

Options:
  -h --help               Show this screen.
  --output=PATH           outfile path, if not given, we just append ".h5" to
                          the input file path
'''
import numpy as np
import h5py
from digicampipe.io.event_stream import calibration_event_stream

from tqdm import tqdm
from docopt import docopt

import numba


@numba.jit
def estimate_arrival_time(adc, thr=0.5):
    '''
    estimate the pulse arrival time, defined as the time the leading edge
    crossed 50% of the maximal height,
    estimated using a simple linear interpolation.

    *Note*
    This method breaks down for very small pulses where noise affects the
    leading edge significantly.
    Typical pixels have a ~2LSB electronics noise level.
    Assuming a leading edge length of 4 samples
    then for a typical pixel, pulses of 40LSB (roughly 7 p.e.)
    should be fine.

    adc: (1296, 50) dtype=uint16 or so
    thr: threshold, 50% by default ... can be played with.

    return:
        arrival_time (1296) in units of time_slices
    '''
    n_pixel = adc.shape[0]
    arrival_times = np.zeros(n_pixel, dtype='f4')

    for pixel_id in range(n_pixel):
        y = adc[pixel_id]
        am = y.argmax()
        y_ = y[:am+1]
        lim = y_[-1] * thr
        foo = np.where(y_ < lim)[0]
        if len(foo):
            start = foo[-1]
            stop = start + 1
            arrival_times[pixel_id] = start + (
                (lim-y_[start]) / (y_[stop]-y_[start])
            )
        else:
            arrival_times[pixel_id] = np.nan
    return arrival_times


@numba.jit
def fill_hist2d(adc, t0, t, histo):
    _range = [[-10, 40], [-0.2, 1.5]]
    _extent = _range[0] + _range[1]
    for pid in range(adc.shape[0]):
        H, xedges, yedges = np.histogram2d(
            t-t0[pid],
            adc[pid],
            bins=(101, 101),
            range=_range
        )
        histo[pid] += H.astype('u2')
    return _extent


def main(outfile_path, input_files=[]):
    events = calibration_event_stream(input_files, max_events=100)
    Rough_factor_between_single_pe_amplitude_and_integral = 21 / 5.8
    histo = None

    for e in tqdm(events):
        adc = e.data.adc_samples
        adc = adc - e.data.digicam_baseline[:, None]

        # I just integrate between sample 10 and 30 to normalize a bit
        # normalizing to maximum_amplitude = 1 is too "sharp"
        integral = adc[:, 10:30].sum(axis=1)
        adc = (
            adc / integral[:, None]
        ) * Rough_factor_between_single_pe_amplitude_and_integral

        t0 = estimate_arrival_time(adc) * 4
        t = np.arange(adc.shape[1]) * 4
        if histo is None:
            histo = np.zeros(
                (adc.shape[0], 101, 101),
                dtype='u2'
            )
        _extent = fill_hist2d(adc, t0, t, histo)

    outfile = h5py.File(outfile_path)
    dset = outfile.create_dataset(
        name='adc_count_histo',
        data=histo,
    )
    dset.attrs['extent'] = _extent


def entry():
    args = docopt(__doc__)
    if '--output' not in args:
        args['--output'] = args['<input_files>'] + '.h5'
    main(
        outfile_path=args['--output'],
        input_files=args['<input_files>'],
    )

if __name__ == '__main__':
    entry()
