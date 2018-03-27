'''
Make a "Bias Curve" or perform a "Rate-scan",
i.e. measure the trigger rate as a function of threshold.

Usage:
  bias_curve_mc <hdf5_file> <outfile>
'''
from docopt import docopt
from digicampipe.calib.camera import filter, r0, random_triggers
from digicampipe.io.save_bias_curve import save_bias_curve
from digicampipe.io.event_stream import event_stream
import numpy as np


if __name__ == '__main__':
    args = docopt(__doc__)

    n_bins = 1024
    thresholds = np.arange(0, 400, 10)
    blinding = True

    data_stream = event_stream(args['<hdf5_file>'])
    data_stream = r0.fill_event_type(data_stream, flag=8)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = r0.fill_trigger_patch(data_stream)
    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = r0.fill_trigger_input_19(data_stream)
    data_stream = save_bias_curve(
        data_stream,
        thresholds=thresholds,
        blinding=blinding,
        output_filename=args['<outfile>']
    )
