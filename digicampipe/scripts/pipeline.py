#!/usr/bin/env python
"""
Run the standard pipeline up to Hillas parameters

Usage:
  digicam-pipeline [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyze
  -o FILE --output=FILE       file where to store the results.
                              [Default: ./hillas.fits]
  --dark=FILE                 File containing the Histogram of
                              the dark analysis
  -v --debug                  Enter the debug mode.
  -c --compute
  -d --display
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --picture_threshold=N       Tailcut primary cleaning threshold
                              [Default: 20]
  --boundary_threshold=N      Tailcut secondary cleaning threshold
                              [Default: 15]
  --parameters=FILE           Calibration parameters file path
  --template=FILE             Pulse template file path
"""
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from ctapipe.core import Field
from ctapipe.io.containers import HillasParametersContainer
from ctapipe.io.serializer import Serializer
from digicampipe.calib.camera import baseline, peak, charge, cleaning, image
from docopt import docopt
from histogram.histogram import Histogram1D

from digicampipe.calib import filter
from digicampipe.image.hillas import compute_alpha, compute_miss
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils import DigiCam
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.visualization.plot import plot_array_camera


class PipelineOutputContainer(HillasParametersContainer):

    local_time = Field(int, 'event time')
    event_id = Field(int, 'event identification number')
    event_type = Field(int, 'event type')

    alpha = Field(float, 'Alpha parameter of the shower')
    miss = Field(float, 'Miss parameter of the shower')
    border = Field(bool, 'Is the event touching the camera borders')


def main(files, max_events, dark_filename, pixel_ids, shift, integral_width,
         debug, hillas_filename, parameters_filename, compute, display,
         picture_threshold, boundary_threshold, template_filename):

    if compute:

        with open(parameters_filename) as file:

            calibration_parameters = yaml.load(file)

        pulse_template = NormalizedPulseTemplate.load(template_filename)

        pulse_area = pulse_template.integral() * u.ns
        ratio = pulse_template.compute_charge_amplitude_ratio(
            integral_width=integral_width, dt_sampling=4)  # ~ 0.24

        gain = np.array(calibration_parameters['gain'])  # ~ 20 LSB / p.e.
        gain_amplitude = gain * ratio

        crosstalk = np.array(calibration_parameters['mu_xt'])
        bias_resistance = 10 * 1E3 * u.Ohm  # 10 kOhm
        cell_capacitance = 50 * 1E-15 * u.Farad  # 50 fF
        geom = DigiCam.geometry

        dark_histo = Histogram1D.load(dark_filename)
        dark_baseline = dark_histo.mean()

        events = calibration_event_stream(files, pixel_id=pixel_ids,
                                          max_events=max_events)
        events = baseline.fill_dark_baseline(events, dark_baseline)
        events = baseline.fill_digicam_baseline(events)
        events = baseline.compute_baseline_shift(events)
        events = baseline.subtract_baseline(events)
        # events = baseline.compute_baseline_std(events, n_events=100)
        events = filter.filter_clocked_trigger(events)
        events = baseline.compute_nsb_rate(events, gain_amplitude,
                                           pulse_area, crosstalk,
                                           bias_resistance, cell_capacitance)
        events = baseline.compute_gain_drop(events, bias_resistance,
                                            cell_capacitance)
        events = peak.find_pulse_with_max(events)
        events = charge.compute_charge(events, integral_width, shift)
        events = charge.compute_photo_electron(events, gains=gain)
        # events = cleaning.compute_cleaning_1(events, snr=3)

        events = cleaning.compute_tailcuts_clean(
            events, geom=geom, overwrite=True,
            picture_thresh=picture_threshold,
            boundary_thresh=boundary_threshold, keep_isolated_pixels=False
        )
        events = cleaning.compute_boarder_cleaning(events, geom,
                                                   boundary_threshold)
        events = cleaning.compute_dilate(events, geom)

        events = image.compute_hillas_parameters(events, geom)

        output_file = Serializer(hillas_filename, mode='w', format='fits')

        data_to_store = PipelineOutputContainer()

        for event in events:

            if debug:

                print(event.hillas)
                print(event.data.nsb_rate)
                print(event.data.gain_drop)
                print(event.data.baseline_shift)
                print(event.data.border)
                plot_array_camera(np.max(event.data.adc_samples, axis=-1))
                plot_array_camera(np.nanmax(
                    event.data.reconstructed_charge, axis=-1))
                plot_array_camera(event.data.cleaning_mask.astype(float))
                plot_array_camera(event.data.reconstructed_number_of_pe)
                plt.show()

            data_to_store.alpha = compute_alpha(event.hillas)
            data_to_store.miss = compute_miss(event.hillas,
                                              data_to_store.alpha)
            data_to_store.local_time = event.data.local_time
            data_to_store.event_type = event.event_type
            data_to_store.event_id = event.event_id
            data_to_store.border = event.data.border

            for key, val in event.hillas.items():

                data_to_store[key] = val

            output_file.add_container(data_to_store)
        output_file.close()

    if display:

        data = Table.read(hillas_filename, format='fits')
        data = data.to_pandas()

        data['local_time'] = pd.to_datetime(data['local_time'])
        data = data.set_index('local_time')
        data = data.dropna()

        for key, val in data.items():

            plt.figure()
            plt.hist(val, bins='auto')
            plt.xlabel(key)

            plt.figure()
            plt.plot(val)
            plt.ylabel(key)

        plt.show()


def entry():

    args = docopt(__doc__)
    files = args['<INPUT>']
    max_events = convert_max_events_args(args['--max_events'])
    dark_filename = args['--dark']
    output = args['--output']
    compute = args['--compute']
    display = args['--display']
    output_path = os.path.dirname(output)

    if not os.path.exists(output_path):
        raise IOError('Path ' + output_path +
                      'for output hillas does not exists \n')
    pixel_ids = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    picture_threshold = float(args['--picture_threshold'])
    boundary_threshold = float(args['--boundary_threshold'])
    shift = int(args['--shift'])
    debug = args['--debug']
    parameters_filename = args['--parameters']
    # args['--min_photon'] = int(args['--min_photon'])
    template_filename = args['--template']
    main(files=files,
         max_events=max_events,
         dark_filename=dark_filename,
         pixel_ids=pixel_ids,
         shift=shift,
         integral_width=integral_width,
         debug=debug,
         parameters_filename=parameters_filename,
         hillas_filename=output,
         compute=compute,
         display=display,
         picture_threshold=picture_threshold,
         boundary_threshold=boundary_threshold,
         template_filename=template_filename
         )


if __name__ == '__main__':
    entry()
