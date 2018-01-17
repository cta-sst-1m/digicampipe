#!/usr/bin/env python
'''
Extract baseline from observations.

    Example call:
    baseline.py ./baseline.npz path/to/SST1M01_20171030.002.fits.fz
    speed is ~100 events/second.
    duration ~2minutes.

Usage:
  baseline.py <baseline_file_path> <files>...

Options:
  -h --help     Show this screen.
'''
from digicampipe.calib.camera import filter, r0
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import save_dark
from digicampipe.io.save_bias_curve import save_bias_curve
from cts_core.camera import Camera
from digicampipe.utils import geometry
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pkg_resources
from os import path
from docopt import docopt

def main(baseline_file_path, files):
    camera_config_file = pkg_resources.resource_filename(
        'digicampipe',
        path.join(
            'tests',
            'resources',
            'camera_config.cfg'
        )
    )

    thresholds = np.arange(0, 400, 10)
    unwanted_patch = [306, 318, 330, 342, 200]
    unwanted_cluster = [200]
    blinding = True

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)
    data_stream = event_stream(
        file_list=files,
        expert_mode=True,
        camera_geometry=digicam_geometry,
        camera=digicam
    )
    data_stream = filter.set_patches_to_zero(
        data_stream,
        unwanted_patch=unwanted_patch)
    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = save_dark(data_stream, baseline_file_path)

    for _ in tqdm(data_stream):
        pass

if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    main(
        baseline_file_path=args['<baseline_file_path>'],
        files=args['<files>']
    )
