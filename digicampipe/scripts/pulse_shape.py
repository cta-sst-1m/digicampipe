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
        y -= y.min()
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


class Histogram2d:

    def __init__(self, shape, range):
        self.histo = np.zeros(shape, dtype='u2')
        self.range = range
        self.extent = self.range[0] + self.range[1]

    def fill(self, time_in_ns, arrival_time_in_ns, data):
        for pixel_id in range(data.shape[0]):
            H, xedges, yedges = np.histogram2d(
                time_in_ns - arrival_time_in_ns[pixel_id],
                data[pixel_id],
                bins=self.histo.shape[1:],
                range=self.range
            )
            self.histo[pixel_id] += H.astype('u2')

    def contents(self):
        return self.histo


class Histogram2dChunked:

    def __init__(self, shape, range, buffer_size=1000):
        self.histo = np.zeros(shape, dtype='u2')
        self.range = range
        self.extent = self.range[0] + self.range[1]

        self.buffer_size = buffer_size
        self.buffer_x = None
        self.buffer_y = None
        self.buffer_counter = 0

    def __reset_buffer(self, x, y):
        self.buffer_x = np.zeros(
            (self.buffer_size, *x.shape),
            dtype=x.dtype
        )
        self.buffer_y = np.zeros(
            (self.buffer_size, *y.shape),
            dtype=y.dtype
        )
        self.buffer_counter = 0

    def fill(self, time_in_ns, arrival_time_in_ns, data):
        '''
        data: (n_pixel, n_samples)
        time_in_ns: (n_samples)
        arrival_time_in_ns: (n_pixel)
        '''
        x = time_in_ns[None, :] - arrival_time_in_ns[:, None]
        y = data
        if self.buffer_counter == self.buffer_size:
            self.__fill_histo_from_buffer()

        if self.buffer_x is None:
            self.__reset_buffer(x, y)

        self.buffer_x[self.buffer_counter] = x
        self.buffer_y[self.buffer_counter] = y
        self.buffer_counter += 1

    def __fill_histo_from_buffer(self):
        if self.buffer_x is None:
            return

        self.buffer_x = self.buffer_x[:self.buffer_counter+1]
        self.buffer_y = self.buffer_y[:self.buffer_counter+1]
        for pixel_id in range(self.buffer_x.shape[1]):
            foo = self.buffer_x[:, pixel_id].flatten()
            bar = self.buffer_y[:, pixel_id].flatten()
            H, xedges, yedges = np.histogram2d(
                foo,
                bar,
                bins=self.histo.shape[1:],
                range=self.range
            )
            self.histo[pixel_id] += H.astype('u2')
        self.buffer_x = None
        self.buffer_y = None
        self.buffer_counter = 0

    def contents(self):
        self.__fill_histo_from_buffer()
        return self.histo


def main(outfile_path, input_files=[]):
    events = calibration_event_stream(input_files)
    Rough_factor_between_single_pe_amplitude_and_integral = 21 / 5.8
    histo = None

    _range = [[-10, 40], [-0.2, 1.5]]

    for e in tqdm(events):
        adc = e.data.adc_samples
        adc = adc - e.data.digicam_baseline[:, None]

        # I just integrate between sample 10 and 30 to normalize a bit
        # normalizing to maximum_amplitude = 1 is too "sharp"
        integral = adc[:, 10:30].sum(axis=1)

        # handling special case .. we say negative integrals make no sense
        # and zero integral simply means there was no pulse at all.
        # so we clip at 1
        integral = integral.clip(1)

        adc = (
            adc / integral[:, None]
        ) * Rough_factor_between_single_pe_amplitude_and_integral

        arrival_time_in_ns = estimate_arrival_time(adc) * 4
        time_in_ns = np.arange(adc.shape[1]) * 4

        # TODO: Would be nice to move this out of the loop
        if histo is None:
            histo = Histogram2dChunked(
                shape=(adc.shape[0], 101, 101),
                range=_range
            )

        histo.fill(time_in_ns, arrival_time_in_ns, adc)

    outfile = h5py.File(outfile_path)
    dset = outfile.create_dataset(
        name='adc_count_histo',
        data=histo.contents(),
        compression='gzip'
    )
    dset.attrs['extent'] = histo.extent


def entry():
    args = docopt(__doc__)
    if args['--output'] is None:
        args['--output'] = args['<input_files>'][0] + '.h5'
    main(
        outfile_path=args['--output'],
        input_files=args['<input_files>'],
    )

if __name__ == '__main__':
    entry()
