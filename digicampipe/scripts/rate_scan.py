"""
Make a "Bias Curve" or perform a "Rate-scan",
i.e. measure the trigger rate as a function of threshold.

Usage:
  digicam-rate-scan [options] [--] <INPUT>...

Options:
  --display                   Display the plots
  --compute                   Computes the trigger rate vs threshold
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
                              [default: ./rate_scan.npz]
  -i INPUT --input=INPUT.     Input files.
  --threshold_start=N         Trigger threshold start
                              [default: 0]
  --threshold_end=N           Trigger threshold end
                              [default: 4095]
  --threshold_step=N          Trigger threshold step
                              [default: 5]
  --n_samples=N               Number of pre-samples used by DigiCam to compute
                              baseline
                              [default: 1024]
"""
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
import fitsio

from digicampipe.calib import filters
from digicampipe.calib import trigger, baseline
from digicampipe.calib.trigger import compute_bias_curve
from digicampipe.io.event_stream import event_stream


def compute(files, output_filename, thresholds, n_samples=1024):

    data_stream = event_stream(files)
    # data_stream = trigger.fill_event_type(data_stream, flag=8)
    data_stream = baseline.fill_baseline_r0(data_stream, n_bins=n_samples)
    data_stream = filters.filter_missing_baseline(data_stream)
    data_stream = trigger.fill_trigger_patch(data_stream)
    data_stream = trigger.fill_trigger_input_7(data_stream)
    data_stream = trigger.fill_trigger_input_19(data_stream)
    output = compute_bias_curve(
        data_stream,
        thresholds=thresholds,
    )

    rate, rate_error, cluster_rate, cluster_rate_error, thresholds = output

    np.savez(file=output_filename, rate=rate, rate_error=rate_error,
             cluster_rate=cluster_rate, cluster_rate_error=cluster_rate_error,
             thresholds=thresholds)

    return output


def entry():
    args = docopt(__doc__)
    input_files = args['<INPUT>']
    output_file = args['--output']
    start = float(args['--threshold_start'])
    end = float(args['--threshold_end'])
    step = float(args['--threshold_step'])
    thresholds = np.arange(start, end + step, step)
    n_samples = int(args['--n_samples'])

    if args['--compute']:
        compute(input_files, output_file, thresholds=thresholds,
                n_samples=n_samples)

    if args['--display']:
        output = np.load(output_file)

        thresholds = output['thresholds']
        rate = output['rate']
        rate_error = output['rate_error']

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.errorbar(thresholds, rate * 1E9, yerr=rate_error * 1E9)
        axes.set_yscale('log')
        axes.set_xlabel('Threshold [LSB]')
        axes.set_ylabel('Rate [Hz]')

        plt.show()


if __name__ == '__main__':
    entry()
