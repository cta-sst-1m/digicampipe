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
from digicampipe import processors as proc


def main(
    input_files,
    dark_baseline_path,
    minimal_number_of_photons,
    output_filename
):

    process = [
        proc.filters.SetPixelsToZero([
            1038, 1039,
            1002, 1003, 1004,
            966, 967, 968,
            930, 931, 932,
            896
        ]),
        proc.baseline.FillBaseline(n_bins=1000),
        proc.filters.FilterEventTypes(flags=[1, 2]),
        proc.filters.FilterMissingBaseline(),
        proc.baseline.R1SubtractBaseline(),
        proc.baseline.R1FillGainDropAndNsb_With_DarkBaseline(
            dark_baseline=np.load(dark_baseline_path)
        ),
        proc.calib.CalibrateToDL1(
            time_integration_options={
                'window_start': 3,
                'window_width': 7,
                'threshold_saturation': np.inf,
                'n_samples': 50,
                'timing_width': 6,
                'central_sample': 11},
            picture_threshold=15,
            boundary_threshold=10,
        ),
        proc.filters.FilterShower(minimal_number_of_photons),
        proc.calib.dlw.CalibrateT0DL2(
            reclean=True,
            shower_distance=200 * u.mm
        ),
        proc.io.HillasToText(output_filename)
    ]

    # proc.run_process(process, input_files)
    for event in event_stream(input_files):
        try:
            for processor in process:
                event = processor(event)
        except proc.SkipEvent:
                continue


if __name__ == '__main__':
    args = docopt(__doc__)

    main(
        input_files=args['<files>'],
        dark_baseline_path=args['--baseline_path'],
        minimal_number_of_photons=int(args['--min_photon']),
        output_filename=args['--outfile_path']
    )
