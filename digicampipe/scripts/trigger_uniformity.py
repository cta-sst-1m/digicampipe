"""
Make a trigger uniformity map
Usage:
  trigger_uniformity.py [options] [--] <INPUT>...

Options:
  --help                        Show this
  --plot=FILE                   path to the output plot. Will show the average
                                over all events of the trigger rate.
                                If set to "show", the plot is displayed and not
                                saved.
                                If set to "none", no plot is done.
                                [Default: show]
  --event_types=LIST            comma separated list of the event type which
                                will be used to calculate the rate:
                                1: patch7 trigger
                                8: clocked trigger
                                By default set to none which keeps all events.
                                [Default: none]
  --disable_bar                 If used, the progress bar is not show while
                                reading files.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.visualization import CameraDisplay
from docopt import docopt

from digicampipe.calib.filters import filter_event_types
from digicampipe.instrument.camera import DigiCam
from digicampipe.instrument.geometry import compute_patch_matrix
from digicampipe.io.event_stream import event_stream
from digicampipe.utils.docopt import convert_text, convert_list_int


def trigger_uniformity(files, plot="show", event_types=None,
                       disable_bar=False):
    events = event_stream(files, disable_bar=disable_bar)
    if event_types is not None:
        events = filter_event_types(events, flags=event_types)
    # patxh matrix is a bool of size n_patch x n_pixel
    patch_matrix = compute_patch_matrix(camera=DigiCam)
    n_patch, n_pixel = patch_matrix.shape
    top7 = np.zeros([n_patch], dtype=np.float32)
    n_event = 0
    for event in events:
        n_event += 1
        tel = event.r0.tels_with_data[0]
        top7 += np.sum(event.r0.tel[tel].trigger_output_patch7, axis=1)
    patches_rate = top7 / n_event
    pixels_rate = patches_rate.reshape([1, -1]).dot(patch_matrix).flatten()
    print('pixels_rate from', np.min(pixels_rate), 'to', np.max(pixels_rate),
          'trigger/event')
    if plot is None:
        return pixels_rate
    fig1 = plt.figure()
    ax = plt.gca()
    display = CameraDisplay(DigiCam.geometry, ax=ax,
                            title='Trigger uniformity')
    display.add_colorbar()
    display.image = pixels_rate
    output_path = os.path.dirname(plot)
    if plot == "show" or not os.path.isdir(output_path):
        if not plot == "show":
            print('WARNING: Path ' + output_path + ' for output trigger ' +
                  'uniformity does not exist, displaying the plot instead.\n')
        plt.show()
    else:
        plt.savefig(plot)
    plt.close(fig1)
    return pixels_rate


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    plot = convert_text(args['--plot'])
    event_types = convert_list_int(args['--event_types'])
    disable_bar = args['--disable_bar']
    trigger_uniformity(files, plot, event_types, disable_bar)


if __name__ == '__main__':
    entry()