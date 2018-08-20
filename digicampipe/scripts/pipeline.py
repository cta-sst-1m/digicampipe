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
  -v --debug                  Enter the debug mode.
  -c --compute
  --display
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --save_figures              Save the plots to the OUTPUT folder
  --parameters=FILE           Calibration parameters file path
"""
from digicampipe.io.event_stream import calibration_event_stream
from ctapipe.io.hdf5tableio import HDF5TableWriter
from digicampipe.visualization import EventViewer
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.calib.camera import baseline, peak, charge, cleaning, image, \
    filter
from digicampipe.utils import DigiCam
import os
import yaml
from docopt import docopt
from histogram.histogram import Histogram1D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from digicampipe.visualization.plot import plot_array_camera
from digicampipe.utils.hillas import correct_alpha_1

def main(files, max_events, dark_filename, pixel_ids, shift, integral_width,
         debug, output_path, parameters_filename, compute, display):
    # Input/Output files

    output_filename = os.path.join(output_path, 'hillas.h5')

    if compute:

        with open(parameters_filename) as file:

            calibration_parameters = yaml.load(file)

        gain = np.array(calibration_parameters['gain'])
        pulse_area = 4
        crosstalk = np.array(calibration_parameters['mu_xt'])
        bias_resistance = 10 * 1E3
        cell_capacitance = 50 * 1E-15
        picture_threshold = 20
        boundary_threshold = 10
        geom = DigiCam.geometry

        dark_histo = Histogram1D.load(dark_filename)
        dark_baseline = dark_histo.mean()

        events = calibration_event_stream(files, pixel_id=pixel_ids,
                                          max_events=max_events, baseline_new=True)
        events = baseline.fill_dark_baseline(events, dark_baseline)
        events = baseline.fill_digicam_baseline(events)
        events = baseline.compute_baseline_shift(events)
        events = baseline.subtract_baseline(events)
        events = baseline.compute_baseline_std(events, n_events=100)
        events = filter.filter_clocked_trigger(events)
        events = baseline.compute_nsb_rate(events, gain, pulse_area, crosstalk,
                                           bias_resistance, cell_capacitance)
        events = baseline.compute_gain_drop(events, bias_resistance,
                                            cell_capacitance)
        events = peak.find_pulse_with_max(events)
        events = charge.compute_charge(events, integral_width, shift)
        events = charge.compute_photo_electron(events, gains=gain)
        events = cleaning.compute_cleaning_1(events, snr=3)
        events = cleaning.compute_tailcuts_clean(events, geom=geom,
                    overwrite=False,
                    picture_thresh=picture_threshold,
                    boundary_thresh=boundary_threshold,
                    keep_isolated_pixels=False)
        events = cleaning.compute_boarder_cleaning(events, geom,
                                                   boundary_threshold)
        events = cleaning.compute_dilate(events, geom)

        events = image.compute_hillas_parameters(events, geom)

        with HDF5TableWriter(output_filename, 'data') as f:

            for event in events:

                if debug:
                    print(event.hillas)
                    plot_array_camera(np.nanmax(event.data.reconstructed_charge,
                                                axis=-1))
                    plot_array_camera(event.data.cleaning_mask.astype(float))
                    plot_array_camera(event.data.reconstructed_number_of_pe)
                    plt.show()

                event.info.type = event.event_type
                event.info.time = event.data.local_time
                event.info.event_id = event.event_id

                f.write('hillas', event.hillas)
                f.write('info', event.info)

    if display:

        data = pd.read_hdf(output_filename, key='data/hillas')
        meta = pd.read_hdf(output_filename, key='data/info')

        plt.figure()
        plt.plot(meta['time'], data['intensity'])

        data = data.dropna()

        plt.figure()

        data = correct_alpha_1(data)
        plt.hist(data['alpha'], bins='auto')

        for key, val in data.items():

            plt.figure()
            plt.hist(val, bins='auto')
            plt.xlabel(key)



        plt.show()


def entry():

    args = docopt(__doc__)
    files = args['<INPUT>']
    max_events = convert_max_events_args(args['--max_events'])
    dark_filename = args['--dark']
    output_path = args['--output']
    compute = args['--compute']
    display = args['--display']

    if not os.path.exists(output_path):
        raise IOError('Path for output does not exists \n')

    pixel_ids = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    debug = args['--debug']
    parameters_filename = args['--parameters']
    # args['--min_photon'] = int(args['--min_photon'])
    main(files=files,
         max_events=max_events,
         dark_filename=dark_filename,
         pixel_ids=pixel_ids,
         shift=shift,
         integral_width=integral_width,
         debug=debug,
         parameters_filename=parameters_filename,
         output_path=output_path,
         compute=compute,
         display=display)


if __name__ == '__main__':

    entry()
