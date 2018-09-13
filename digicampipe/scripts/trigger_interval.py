#!/usr/bin/env python
"""
Do an histogram of the interval between 2 triggers

Usage:
  digicam-raw [options] [--] <INPUT>...

Options:
  -h --help                   Show this help.
  --max_events=N              Maximum number of events to analyse
  -o FILE --output=FILE.      File where to store the results.
                              [Default: ./trigger_interval.pk]
  -p --plot=FILE              path to the output plot history of rate.
                              Use "show" to open an interactive plot instead
                              of creating a file.
                              [Default: show]
"""
import os
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_max_events_args


def compute(files, max_events, filename):
    if os.path.exists(filename) and len(files) == 0:
        dt_histo = Histogram1D.load(filename)
        return dt_histo
    else:
        events = calibration_event_stream(files, max_events=max_events)
        dt_histo = Histogram1D(
            data_shape=(9,),
            bin_edges=np.logspace(2, 9, 140),  # in ns
            # bin_edges=np.arange(150, 400, 4),  # in ns
        )
        previous_time = np.zeros(9) * np.nan
        #skipped = np.zeros(9)
        for event in events:
            typ = event.event_type
            local_time = event.data.local_time # in ns
            dt = local_time - previous_time[typ] # in ns
            if np.isfinite(previous_time[typ]):
                dt_histo.fill(dt, indices=(event.event_type,))
            previous_time[typ] = local_time
        #for typ in range(9):
        #    if skipped[typ] > 0:
        #        print('skipped', skipped[typ], 'events for type', typ)
        dt_histo.save(filename)
        print(filename, 'saved')
        return dt_histo


def entry(files, max_events, raw_histo_filename, plot):
    output_path = os.path.dirname(raw_histo_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dt_histo = compute(files, max_events, raw_histo_filename)
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111)
    for typ in range(9):
        if not np.isfinite(dt_histo.mean(typ)):
            continue
        dt_histo.draw(index=(typ,), axis=axis, log=True, legend=False,
                      x_label='$\Delta t_{trig} [ns]$', label='type '+str(typ))
    axis.set_xscale('log')
    if plot.lower() == "show":
        plt.show()
    else:
        path = os.path.dirname(plot)
        if not os.path.exists(path):
            os.makedirs(path)
        figure.savefig(plot)
        print(plot, 'saved')
    plt.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    max_events = convert_max_events_args(args['--max_events'])
    raw_histo_filename = args['--output']
    plot = args['--plot']
    entry(files, max_events, raw_histo_filename, plot)
