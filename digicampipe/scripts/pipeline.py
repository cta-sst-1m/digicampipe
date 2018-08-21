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
  --picture_threshold=N       Tailcut primary cleaning threshold
                              [Default: 20]
  --boundary_threshold=N      Tailcut secondary cleaning threshold
                              [Default: 15]
  --parameters=FILE           Calibration parameters file path
"""
from digicampipe.io.event_stream import calibration_event_stream
from ctapipe.io.serializer import Serializer
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
         debug, output_path, parameters_filename, compute, display,
         picture_threshold, boundary_threshold):
    # Input/Output files

    hillas_filename = os.path.join(output_path, 'hillas.fits')
    meta_filename = os.path.join(output_path, 'meta.fits')

    if compute:

        with open(parameters_filename) as file:

            calibration_parameters = yaml.load(file)

        gain = np.array(calibration_parameters['gain'])
        pulse_area = 4
        crosstalk = np.array(calibration_parameters['mu_xt'])
        bias_resistance = 10 * 1E3
        cell_capacitance = 50 * 1E-15
        geom = DigiCam.geometry

        dark_histo = Histogram1D.load(dark_filename)
        dark_baseline = dark_histo.mean()

        events = calibration_event_stream(files, pixel_id=pixel_ids,
                                          max_events=max_events,
                                          baseline_new=True)
        events = baseline.fill_dark_baseline(events, dark_baseline)
        events = baseline.fill_digicam_baseline(events)
        events = baseline.compute_baseline_shift(events)
        events = baseline.subtract_baseline(events)
        # events = baseline.compute_baseline_std(events, n_events=100)
        events = filter.filter_clocked_trigger(events)
        events = baseline.compute_nsb_rate(events, gain, pulse_area, crosstalk,
                                           bias_resistance, cell_capacitance)
        events = baseline.compute_gain_drop(events, bias_resistance,
                                            cell_capacitance)
        events = peak.find_pulse_with_max(events)
        events = charge.compute_charge(events, integral_width, shift)
        events = charge.compute_photo_electron(events, gains=gain)
        # events = cleaning.compute_cleaning_1(events, snr=3)
        events = cleaning.compute_tailcuts_clean(events, geom=geom,
                                                 overwrite=True,
                                                 picture_thresh=
                                                 picture_threshold,
                                                 boundary_thresh=
                                                 boundary_threshold,
                                                 keep_isolated_pixels=False)
        events = cleaning.compute_boarder_cleaning(events, geom,
                                                   boundary_threshold)
        events = cleaning.compute_dilate(events, geom)

        events = image.compute_hillas_parameters(events, geom)

        # with HDF5TableWriter(output_filename, 'data') as f:
        # with Serializer(output_filename, mode='w', format='fits') as f:
        data = Serializer(hillas_filename, mode='w',
                          format='fits')
        meta = Serializer(meta_filename, mode='w',
                          format='fits')

        for event in events:

            if debug:

                print(event.hillas)
                print(event.data.nsb_rate)
                plot_array_camera(np.max(event.data.adc_samples, axis=-1))
                plot_array_camera(np.nanmax(
                    event.data.reconstructed_charge, axis=-1))
                plot_array_camera(event.data.cleaning_mask.astype(float))
                plot_array_camera(event.data.reconstructed_number_of_pe)
                plt.show()

            event.info.type = event.event_type
            event.info.time = event.data.local_time
            event.info.event_id = event.event_id

            data.add_container(event.hillas)
            meta.add_container(event.info)
        data.close()
        meta.close()

    if display:

        from astropy.table import Table
        data = Table.read(hillas_filename, format='fits')
        data = data.to_pandas()

        meta = Table.read(meta_filename, format='fits')
        meta = meta.to_pandas()

        data = pd.concat([data, meta], axis=1)

        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.dropna()

        data = correct_alpha_1(data)

        plt.figure()
        plt.plot(data['intensity'])
        plt.ylabel('intensity')

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
    picture_threshold = float(args['--picture_threshold'])
    boundary_threshold = float(args['--boundary_threshold'])
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
         display=display,
         picture_threshold=picture_threshold,
         boundary_threshold=boundary_threshold)


if __name__ == '__main__':

    entry()
