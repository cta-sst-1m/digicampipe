#!/usr/bin/env python
'''
Extract baseline from observations.

    Example call:
    baseline.py ./baseline.npz path/to/SST1M01_20171030.002.fits.fz

Usage:
  baseline.py <baseline_file_path> <files>...

Options:
  -h --help     Show this screen.
'''
from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import save_dark
from digicampipe.utils import Camera
from tqdm import tqdm
from docopt import docopt


def main(baseline_file_path, files):

    unwanted_patch = [306, 318, 330, 342, 200]

    digicam = Camera()
    data_stream = event_stream(
        file_list=files,
        expert_mode=True,
        camera=digicam,
        camera_geometry=digicam.geometry
    )
    data_stream = filter.set_patches_to_zero(
        data_stream,
        unwanted_patch=unwanted_patch)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = save_dark(data_stream, baseline_file_path)

    for _ in tqdm(data_stream):
        pass


if __name__ == "__main__":
    args = docopt(__doc__)
    main(
        baseline_file_path=args['<baseline_file_path>'],
        files=args['<files>']
    )
