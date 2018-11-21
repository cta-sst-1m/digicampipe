"""
Make a "Bias Curve" or perform a "Rate-scan",
i.e. measure the trigger rate as a function of threshold.

Usage:
  digicam-rate-scan [options] [--] <INPUT>...

Options:
  --display                   Display the plots
  --compute                   Computes the trigger rate vs threshold
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
                              [default: ./rate_scan.fits]
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
  --figure_path=OUTPUT        Figure path
                              [default: None]
"""
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
import fitsio
import pandas as pd

from digicampipe.calib import filters
from digicampipe.calib import trigger, baseline
from digicampipe.calib.trigger import compute_bias_curve
from digicampipe.io.event_stream import event_stream
from digicampipe.io.containers import CameraEventType


def compute(files, output_filename, thresholds, n_samples=1024):

    thresholds = thresholds.astype(float)

    data_stream = event_stream(files)
    # data_stream = trigger.fill_event_type(data_stream, flag=8)
    data_stream = filters.filter_event_types(data_stream,
                                             flags=CameraEventType.INTERNAL)
    data_stream = baseline.fill_baseline_r0(data_stream, n_bins=n_samples)
    data_stream = filters.filter_missing_baseline(data_stream)
    data_stream = trigger.fill_trigger_patch(data_stream)
    data_stream = trigger.fill_trigger_input_7(data_stream)
    data_stream = trigger.fill_trigger_input_19(data_stream)
    output = compute_bias_curve(
        data_stream,
        thresholds=thresholds,
    )

    rate, rate_error, cluster_rate, cluster_rate_error, thresholds, \
    start_event_id, end_event_id, start_event_time, end_event_time = output

    with fitsio.FITS(output_filename, mode='rw', clobber=True) as f:

        f.write([np.array([start_event_id, end_event_id]),
                 np.array([start_event_time, end_event_time])],
                extname='meta',
                names=['event_id', 'time'])
        f.write(thresholds, extname='threshold', compress='gzip')
        f.write([rate, rate_error], extname='camera', names=['rate', 'error'],
                compress='gzip')
        f.write([cluster_rate, cluster_rate_error], names=['rate', 'error'],
                extname='cluster',
                compress='gzip')

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
    figure_path = args['--figure_path']
    figure_path = None if figure_path == 'None' else figure_path

    if args['--compute']:
        compute(input_files, output_file, thresholds=thresholds,
                n_samples=n_samples)

    if args['--display'] or figure_path is not None:

        with fitsio.FITS(output_file, 'r') as f:

            meta = f['meta']
            id = meta['event_id'].read()
            time = meta['time'].read()
            start_id, end_id = id
            start_time, end_time = time
            thresholds = f['threshold'].read()
            camera_rate = f['camera']['rate'].read()
            camera_rate_error = f['camera']['error'].read()
            cluster_rate = f['cluster']['rate'].read()
            cluster_rate_error = f['cluster']['error'].read()

        start_time = pd.to_datetime(int(start_time), utc=True)
        end_time = pd.to_datetime(int(end_time), utc=True)

        start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.errorbar(thresholds, camera_rate * 1E9,
                      yerr=camera_rate_error * 1E9, marker='o', color='k',
                      label='Start time : {}\nEnd time   : {}\nEvent ID :'
                            ' ({}, {})'.format(start_time, end_time, start_id,
                                               end_id))
        axes.set_yscale('log')
        axes.set_xlabel('Threshold [LSB]')
        axes.set_ylabel('Trigger rate [Hz]')
        axes.legend(loc='best')

        if args['--display']:

            plt.show()

        if figure_path is not None:

            fig.savefig(figure_path)


if __name__ == '__main__':
    entry()
