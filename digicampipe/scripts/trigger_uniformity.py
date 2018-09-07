"""
Make a trigger uniformity map
Usage:
  trigger_uniformity.py [options] [--] <INPUT>...

Options:
  --help                        Show this
  --plot=FILE                   path to the output plot. Will show the average
                                over all events of the trigger rate.
                                If set to none, the plot is displayed and not
                                saved.
                                [Default: none]
  --event_type=LIST             comma separated list of the event type which
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
from digicampipe.io.event_stream import event_stream
from digicampipe.calib.camera.filter import filter_event_types
import os
from digicampipe.utils.geometry import compute_patch_matrix
from ctapipe.visualization import CameraDisplay


def entry(files, plot, event_type='none'):
    events = event_stream(files)
    if event_type is None or event_type == 'none' or event_type == 'None':
        pass
    else:
        flags = [int(flag) for flag in event_type.strip(',').split(',')]
        events = filter_event_types(events, flags=flags)
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
    print(pixels_rate)
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
    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    plot = args['--plot']
    event_type = args['--event_type']
    entry(files, plot, event_type)
