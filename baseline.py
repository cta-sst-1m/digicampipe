#!/usr/bin/env python
'''
Extract baseline from observations.

    Example call:
    baseline.py path/to/outfile.npz path/to/SST1M01_20171030.002.fits.fz

    baseline.py --unwanted_pixels=1,2,3 \
        path/to/outfile.npz \
        ../sst1m_crab/SST1M01_20171030.002.fits.fz

Usage:
  baseline.py [options] <baseline_file_path> <files>...

Options:
  -h --help     Show this screen.
  --unwanted_pixels=<integers>   list of integers with commas [default: ]
'''
import numpy as np
from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import AdcSampleStatistics
from tqdm import tqdm
from docopt import docopt


def estimate_baseline(files, unwanted_pixels=[]):
    adc_sample_stats = AdcSampleStatistics()

    data_stream = event_stream(file_list=files)
    data_stream = filter.set_pixels_to_zero(
        data_stream,
        unwanted_pixels=unwanted_pixels)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = adc_sample_stats(data_stream)

    for _ in tqdm(data_stream):
        pass

    # here I have the results from the analysis in hand.
    # the point is: I can print, store or plot them here,
    # without modifying the function that calculated them.
    # I can even return them:
    return adc_sample_stats
    # This way it is possible to test an analysis


def main(baseline_file_path, files, unwanted_pixels=[]):

    baseline = estimate_baseline(
        files=files,
        unwanted_pixels=unwanted_pixels
    )

    np.savez(
        baseline_file_path,
        baseline=baseline.mean,
        standard_deviation=baseline.std
    )

if __name__ == "__main__":
    args = docopt(__doc__)
    args['--unwanted_pixels'] = [
        int(x) for x in args['--unwanted_pixels'].split(',') if x]
    main(
        baseline_file_path=args['<baseline_file_path>'],
        files=args['<files>'],
        unwanted_pixels=args['--unwanted_pixels']
    )
