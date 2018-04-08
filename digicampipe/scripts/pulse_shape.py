#!/usr/bin/env python
'''
Reconstruct the pulse template

Usage:
  pulse_shape.py [options] [FILE] [INPUT ...]

Options:
  -h --help               Show this screen.
  --max_events=N          Maximum number of events to analyse
  -o FILE --output=FILE.  Output file.
  -i INPUT --input=INPUT. Input files.
  -c --compute            Compute the data.
  -f --fit                Fit.
  -d --display            Display.
  -v --debug              Enter the debug mode.
  -p --pixel=<PIXEL>      Give a list of pixel IDs.
'''
import os
from docopt import docopt
from tqdm import tqdm
from schema import Schema, And, Or, Use, SchemaError
import numpy as np
import matplotlib.pyplot as plt

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.io.containers_calib import CalibrationContainer


def entry():

    args = docopt(__doc__)

    input_files = args['FILE']
    max_events = args['--max_events']
    max_events = max_events if max_events is None else str(max_events)

    container = CalibrationContainer()

    events = calibration_event_stream(input_files, telescope_id=1,
                                      max_events=max_events)

    for count, event in enumerate(events):

        adc_samples = event.data.adc_samples

        if count == 0:

            pulse_template = np.zeros(adc_samples.shape)

        pulse_template += adc_samples

        plt.figure()
        plt.plot(adc_samples[0])
        plt.show()

    pulse_template /= count

    plt.figure()
    plt.plot(pulse_template[0])
    plt.show()


if __name__ == '__main__':
    entry()

