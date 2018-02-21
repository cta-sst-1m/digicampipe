#!/usr/bin/env python
'''

Example:
  ./pipeline_crab.py \
  --baseline_path=../sst1m_crab/dark.npz \
  --outfile_path=./hillas_output.txt \
  ../sst1m_crab/SST1M01_20171030.01*

Usage:
  pipeline_crab.py [options] <files>...


Options:
  -h --help     Show this screen.
  --display     Display rather than output data
  -o <path>, --outfile_path=<path>   path to the output file
  -b <path>, --baseline_path=<path>  path to baseline file usually called "dark.npz"
  --min_photon <int>     Filtering on big showers [default: 20]
'''
from digicampipe.io.event_stream import event_stream
import numpy as np
import astropy.units as u
from docopt import docopt


from digicampipe.processors.filter import (
    SetPixelsToZero,
    FilterEventTypes,
    FilterMissingBaseline,
    FilterShower
)
from digicampipe.processors.baseline import FillBaseline
from digicampipe.processors.calib import (
    R1SubtractBaseline,
    R1FillGainDropAndNsb,
    R1FillGainDropAndNsb_With_DarkBaseline,
)
from digicampipe.processors.calib.dl1 import CalibrateToDL1
from digicampipe.processors.calib.dl2 import CalibrateT0DL2
from digicampipe.processors.io import HillasToText


def main(args):

    # Noisy pixels not taken into account in Hillas
    pixel_not_wanted = [
        1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    additional_mask = np.ones(1296, dtype=bool)
    additional_mask[pixel_not_wanted] = False

    process = [
        SetPixelsToZero(pixel_not_wanted),
        FillBaseline(n_bins=1000),
        FilterEventTypes(flags=[1, 2]),
        FilterMissingBaseline(),

        # These 2 were: calibrate_to_r1  before.
        R1SubtractBaseline(),
        R1FillGainDropAndNsb_With_DarkBaseline(
            dark_baseline=np.load(args['--baseline_path'])),
        CalibrateToDL1(
            time_integration_options={
                'window_start': 3,
                'window_width': 7,
                'threshold_saturation': np.inf,
                'n_samples': 50,
                'timing_width': 6,
                'central_sample': 11},
            picture_threshold=15,
            boundary_threshold=10,
            additional_mask=additional_mask,
        ),
        FilterShower(min_photon=args['--min_photon']),
        CalibrateT0DL2(reclean=True, shower_distance=200 * u.mm),
        HillasToText(output_filename=args['--outfile_path'])
    ]

    for event in event_stream(args['<files>']):
        for processor in process:
            event = processor(event)
            if event is None:
                continue


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    args['--min_photon'] = int(args['--min_photon'])
    main(args)
