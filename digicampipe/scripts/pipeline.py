#!/usr/bin/env python
"""
Run the standard pipeline up to Hillas parameters

Usage:
  digicam-pipeline [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyze
  -o OUTPUT --output=OUTPUT   Folder where to store the results.
  --dark=FILE                 File containing the Histogram of
                              the dark analysis
  -c --compute                Compute the data.
  -f --fit                    Fit.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --save_figures              Save the plots to the OUTPUT folder
  --n_samples=N               Number of samples in readout window
  --parameters=FILE           Calibration parameters file path
"""
from digicampipe.calib.camera import filter, r1, random_triggers, dl0, dl2, dl1
from digicampipe.io.event_stream import calibration_event_stream
from cts_core.utils import Camera
from digicampipe.io.save_hillas import save_hillas_parameters_in_text
from digicampipe.visualization import EventViewer
from digicampipe.utils import utils
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.calib.camera import baseline, peak, charge

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from docopt import docopt
from histogram.histogram import Histogram1D


def main(files, max_events, dark_filename, pixel_ids, shift, integral_width,
         n_samples, debug):
    # Input/Output files

    dark_histo = Histogram1D.load(dark_filename)
    dark_baseline = dark_histo.mean()



    events = calibration_event_stream(files, pixel_id=pixel_ids,
                                      max_events=max_events)
    events = baseline.fill_dark_baseline(events, dark_baseline)
    events = baseline.fill_digicam_baseline(events)
    events = baseline.compute_baseline_shift(events)
    events = baseline.subtract_baseline(events)
    events = baseline.compute_nsb_rate(events, gain, pulse_area, crosstalk,
                                       bias_resistance, cell_capacitance)
    events = baseline.compute_gain_drop(events, bias_resistance,
                                        cell_capacitance)
    events = peak.find_pulse_with_max(events)
    events = charge.compute_charge(events, integral_width, shift)
    events = charge.compute_photo_electron(events, gains=gain)


    # Image cleaning configuration
    picture_threshold = 15
    boundary_threshold = 10
    shower_distance = 200 * u.mm

    # Define the event stream
    data_stream = event_stream(args['<files>'], camera=digicam)
    # Clean pixels
    data_stream = filter.set_pixels_to_zero(data_stream,
                                            unwanted_pixels=pixel_not_wanted)
    # Compute baseline with clocked triggered events
    # (sliding average over n_bins)
    data_stream = random_triggers.fill_baseline_r0(data_stream,
                                                   n_bins=n_bins)
    # Stop events that are not triggered by DigiCam algorithm
    # (end of clocked triggered events)
    data_stream = filter.filter_event_types(data_stream, flags=[1, 2])
    # Do not return events that have not the baseline computed
    # (only first events)
    data_stream = filter.filter_missing_baseline(data_stream)

    # Run the r1 calibration (i.e baseline substraction)
    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)
    # Run the dl1 calibration (compute charge in photons + cleaning)
    data_stream = dl1.calibrate_to_dl1(data_stream,
                                       time_integration_options,
                                       additional_mask=additional_mask,
                                       picture_threshold=picture_threshold,
                                       boundary_threshold=boundary_threshold)
    # Return only showers with total number of p.e. above min_photon
    data_stream = filter.filter_shower(
        data_stream, min_photon=args['--min_photon'])
    # Run the dl2 calibration (Hillas)
    data_stream = dl2.calibrate_to_dl2(
        data_stream, reclean=reclean, shower_distance=shower_distance)

    if args['--display']:

        with plt.style.context('ggplot'):
            display = EventViewer(data_stream)
            display.draw()
    else:
        save_hillas_parameters_in_text(
            data_stream=data_stream, output_filename=args['--outfile_path'])


def entry():

    args = docopt(__doc__)
    print(args)
    files = args['<INPUT>']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['--output']

    if not os.path.exists(output_path):
        raise IOError('Path for output does not exists \n')

    pixel_id = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    args['--min_photon'] = int(args['--min_photon'])
    main(args)

if __name__ == '__main__':

