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
import h5py
import numpy as np
from docopt import docopt
from tqdm import tqdm

from digicampipe.calib.time import estimate_arrival_time
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.hist2d import Histogram2dChunked


def main(outfile_path, input_files=[]):
    events = calibration_event_stream(input_files)
    Rough_factor_between_single_pe_amplitude_and_integral = 21 / 5.8
    histo = None

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
                range=[[-10, 40], [-0.2, 1.5]]
            )

        histo.fill(
            x=time_in_ns[None, :] - arrival_time_in_ns[:, None],
            y=adc
        )

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
