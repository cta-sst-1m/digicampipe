"""
Make a trigger uniformity map
Usage:
  trigger_uniformity.py [options] [--] <INPUT>...

Options:
  --help                        Show this
  --plot=FILE                   path to the output plot. Will show the average
                                over all events of the trigger rate.
                                If set to none, th eplot is displayed and not
                                saved.
                                [Default: none]
  --event_type=FILE             comma separated list of the event type which
                                will be used to calculate the rate:
                                1: patch7 trigger
                                8: clocked trigger
                                By default set to none which keeps all events.
                                [Default: none]
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
from digicampipe.utils import DigiCam
from digicampipe.io.event_stream import event_stream, add_slow_data
import os


def entry(files, plot, even_type='none'):
    input_path = os.path.dirname(files[0])
    aux_basepath = input_path.replace('/raw', '/aux')

    events = event_stream(files)
    events = add_slow_data(events, basepath=aux_basepath)

    top7 = None
    for event in events:
        top7 = event.trigger_output_patch7
        print(top7)
        continue

    fig1 = plt.figure()
    plt.plot()
    plt.plot(top7)
    plt.ylabel('rate [Hz]')
    output_path = os.path.dirname(plot)
    if plot == "show" or not os.path.isdir(output_path):
        if not os.path.isdir(output_path):
            print('WARNING: Path ' + output_path + 'for output trigger ' +
                  'uniformity does not exist, displaying the plot instead.\n')
        plt.show()
    else:
        plt.savefig(plot)
    plt.close(fig1)
    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    plot = args['--plot']
    event_type = args['--event_type']
    entry(files, plot, event_type)
