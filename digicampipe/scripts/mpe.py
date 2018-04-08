#!/usr/bin/env python
'''
Do the Multiple Photoelectron anaylsis

Usage:
  mpe.py [options] [FILE] [INPUT ...]

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import scipy
import pandas as pd
from scipy.ndimage.filters import convolve1d

import peakutils
from iminuit import Minuit, describe
from probfit import Chi2Regression

from ctapipe.io import HDF5TableWriter, HDF5TableReader
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.pdf import gaussian, single_photoelectron_pdf
from digicampipe.utils.exception import PeakNotFound
from digicampipe.io.containers_calib import SPEResultContainer, CalibrationHistogramContainer
from histogram.histogram import Histogram1D
from digicampipe.utils.utils import get_pulse_shape


def entry():
    args = docopt(__doc__)


if __name__ == '__main__':
    entry()
