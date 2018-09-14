"""
plot event_id function of the time
Usage:
  plot_event_id_vs_time [options] [--] <INPUT>...

Options:
  --help                        Show this help.
  --event_number_min=INT        path to histogram of the dark files.
                                [Default: none]
  --event_number_max=INT        Calibration parameters file path.
                                [Default: none]
  --plot=FILE                   Path of the image to be created.
                                Use "show" to open an interactive plot instead
                                of creating a file.
                                Use "none" to skip the plot.
                                [Default: show]
"""
from docopt import docopt
import matplotlib.pyplot as plt
from digicampipe.io.event_stream import event_stream
import numpy as np
import os


def entry(files, event_number_min, event_number_max, plot):
    events = event_stream(files)
    events_id = []
    events_ts = []
    for i, event in enumerate(events):
        tel = event.r0.tels_with_data[0]
        clock_ns = event.r0.tel[tel].local_camera_clock
        event_id = event.r0.tel[tel].camera_event_number
        if event_number_min != "none" and event_id <= int(event_number_min):
            continue
        if  event_number_max != "none" and event_id > int(event_number_max):
            continue
        events_ts.append(clock_ns)
        events_id.append(event_id)
    events_ts = np.array(events_ts)
    if plot.lower() != "none":
        fig1 = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(events_ts-events_ts[0], events_id, '.')
        plt.xlabel('$t [ns]$')
        plt.ylabel('event_id')
        plt.subplot(2, 2, 2)
        plt.plot(np.diff(events_ts), events_id[1:], '.')
        plt.xlim(150, 400)
        plt.xlabel('$\Delta t [ns]$')
        plt.ylabel('event_id')
        plt.subplot(2, 2, 4)
        plt.hist(np.diff(events_ts), np.arange(150, 400, 4))
        plt.xlim(150, 400)
        plt.xlabel('$\Delta t [ns]$')
        plt.ylabel('# of events')
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
    event_number_min = args['--event_number_min']
    event_number_max =args['--event_number_max']
    plot = args['--plot']
    print(files)
    print(event_number_min)
    print(plot)
    entry(files, event_number_min, event_number_max, plot)
