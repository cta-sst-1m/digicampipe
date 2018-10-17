"""
plot event_id function of the time for the selected events
Usage:
  plot_event_id_vs_time [options] [--] <INPUT>...

Options:
  --help                        Show this help.
  --event_id_start=INT          minimum event id to plot. If none, no
                                minimum id is considered.
                                [Default: none]
  --event_id_end=INT            maximum event id to plot. If none, no
                                maximum id is considered.
                                [Default: none]
  --plot=FILE                   Path of the image to be created.
                                Use "show" to open an interactive plot instead
                                of creating a file.
                                Use "none" to skip the plot.
                                [Default: show]
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas import to_datetime

from digicampipe.io.event_stream import event_stream
from digicampipe.utils.docopt import convert_text


def entry(files, event_id_start, event_id_end, plot):
    events = event_stream(
        files,
        event_id_range=(int(event_id_start), int(event_id_end))
    )
    events_id = []
    events_ts = []
    baselines_mean = []
    for i, event in enumerate(events):
        tel = event.r0.tels_with_data[0]
        clock_ns = event.r0.tel[tel].local_camera_clock
        event_id = event.r0.tel[tel].camera_event_number
        baseline_mean = np.mean(event.r0.tel[tel].digicam_baseline)
        events_ts.append(clock_ns)
        events_id.append(event_id)
        baselines_mean.append(baseline_mean)
    events_ts = np.array(events_ts)
    events_id = np.array(events_id)
    baselines_mean = np.array(baselines_mean)
    order = np.argsort(events_ts)
    events_ts = events_ts[order]
    events_id = events_id[order]
    baselines_mean = baselines_mean[order]
    if plot.lower() is not None:
        print('plotted with respect to t=', to_datetime(events_ts[0]))
        fig1 = plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot((events_ts-events_ts[0]) * 1e-9, events_id, '.')
        plt.xlabel('$t [s]$')
        plt.ylabel('event_id')
        plt.subplot(2, 2, 2)
        plt.hist(np.diff(events_ts), np.arange(150, 400, 4))
        plt.xlim(150, 400)
        plt.xlabel('$\Delta t [ns]$')
        plt.ylabel('# of events')
        plt.subplot(2, 1, 2)
        plt.plot((events_ts-events_ts[0]) * 1e-9, baselines_mean, '.')
        plt.xlabel('$t [s]$')
        plt.ylabel('mean baseline [LSB]')
        plt.tight_layout()
        if plot == "show":
            plt.show()
        else:
            output_path = os.path.dirname(plot)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(plot)
        plt.close(fig1)
    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    event_id_start = args['--event_id_start']
    event_id_end = args['--event_id_end']
    plot = convert_text(args['--plot'])
    entry(files, event_id_start, event_id_end, plot)
