'''
Usage:
  dg_bias_curve_mc [options] <filename> <outfilename>

Options:
  --nbins INT  number of bins [default: 1024]
  --blinding   switch on blinding
'''
from docopt import docopt
from digicampipe.calib.camera import filter, r0, random_triggers
from digicampipe.io.save_bias_curve import save_bias_curve
from digicampipe.io.event_stream import event_stream
import numpy as np


def entry():
    args = docopt(__doc__)

    thresholds = np.arange(0, 400, 10)

    data_stream = event_stream(args['<filename>'])
    data_stream = r0.fill_event_type(data_stream, flag=8)
    data_stream = random_triggers.fill_baseline_r0(
        data_stream,
        n_bins=int(args['--nbins']))
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = r0.fill_trigger_patch(data_stream)
    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = r0.fill_trigger_input_19(data_stream)
    data_stream = save_bias_curve(
        data_stream,
        thresholds=thresholds,
        blinding=args['--blinding'],
        output_filename=args['<outfilename>']
    )
