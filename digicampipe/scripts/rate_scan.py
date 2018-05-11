'''
Make a "Bias Curve" or perform a "Rate-scan",
i.e. measure the trigger rate as a function of threshold.

Usage:
  rate_scan [options] <file> <outfile>

Options:
  --display   Display the plots
'''
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt

from digicampipe.calib.camera import filter, r0, random_triggers
from digicampipe.io.save_bias_curve import compute_bias_curve
from digicampipe.io.event_stream import event_stream


def entry():

    args = docopt(__doc__)
    main(args['<file>'], args['<outfile>'], args['--display'])


def main(files, output_filename, display=False):

    n_bins = 1024
    thresholds = np.arange(0, 400, 10)
    blinding = True

    data_stream = event_stream(files)
    data_stream = r0.fill_event_type(data_stream, flag=8)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=n_bins)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = r0.fill_trigger_patch(data_stream)
    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = r0.fill_trigger_input_19(data_stream)
    output = compute_bias_curve(
        data_stream,
        thresholds=thresholds,
        blinding=blinding,
    )

    rate, rate_error, cluster_rate, cluster_rate_error, thresholds = output

    np.savez(file=output_filename, rate=rate, rate_error=rate_error,
             cluster_rate=cluster_rate, cluster_rate_error=cluster_rate_error,
             thresholds=thresholds)

    if display:

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.errorbar(thresholds, rate * 1E9, yerr=rate_error * 1E9)
        axes.set_yscale('log')
        axes.set_xlabel('Threshold [LSB]')
        axes.set_ylabel('Rate [Hz]')

        plt.show()


if __name__ == '__main__':

    entry()
