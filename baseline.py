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
from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import save_dark
from tqdm import tqdm
from docopt import docopt


def main(baseline_file_path, files, unwanted_pixels=[]):

    data_stream = event_stream(files)
    data_stream = filter.set_pixels_to_zero(
        data_stream,
        unwanted_pixels=unwanted_pixels)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = save_dark(data_stream, baseline_file_path)

    for _ in tqdm(data_stream):
        pass


if __name__ == "__main__":
    args = docopt(__doc__)
    args['--unwanted_pixels'] = [
        int(x) for x in args['--unwanted_pixels'].split(',') if x]
    main(
        baseline_file_path=args['<baseline_file_path>'],
        files=args['<files>'],
        unwanted_pixels=args['--unwanted_pixels']
    )
