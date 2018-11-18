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
  -p --bad_pixels=LIST        Give a list of bad pixel IDs.
                              If "none", the bad pixels will be deduced from
                              the parameter file specified with --parameters.
                              [default: none]
  --saturation_threshold=N    Threshold in LSB at which the pulse amplitude is
                              considered as saturated.
                              [default: 3000]
  --threshold_pulse=N         A threshold to which the integration of the pulse
                              is defined for saturated pulses.
                              [default: 0.1]
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --picture_threshold=N       Tailcut primary cleaning threshold
                              [Default: 30]
  --boundary_threshold=N      Tailcut secondary cleaning threshold
                              [Default: 15]
  --parameters=FILE           Calibration parameters file path
  --template=FILE             Pulse template file path
  --disable_bar               If used, the progress bar is not show while
                              reading files.
"""
import os
import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from ctapipe.core import Field
from ctapipe.io.containers import HillasParametersContainer
from ctapipe.io.serializer import Serializer
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from histogram.histogram import Histogram1D

from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.calib import baseline, peak, charge, cleaning, image, tagging
from digicampipe.calib import filters
from digicampipe.instrument.camera import DigiCam
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, convert_list_int, \
    convert_text, convert_float
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.visualization.plot import plot_array_camera
from digicampipe.image.hillas import compute_alpha, compute_miss, \
    correct_alpha_3


class PipelineOutputContainer(HillasParametersContainer):
    local_time = Field(int, 'event time')
    event_id = Field(int, 'event identification number')
    event_type = Field(int, 'event type')

    alpha = Field(float, 'Alpha parameter of the shower')
    miss = Field(float, 'Miss parameter of the shower')
    border = Field(bool, 'Is the event touching the camera borders')
    burst = Field(bool, 'Is the event during a burst')
    saturated = Field(bool, 'Is any pixel signal saturated')


def main_pipeline(
        files, max_events, dark_filename, integral_width,
        debug, hillas_filename, parameters_filename,
        picture_threshold, boundary_threshold, template_filename,
        saturation_threshold, threshold_pulse,
        bad_pixels=None, disable_bar=False
):
    with open(parameters_filename) as file:
        calibration_parameters = yaml.load(file)
    if bad_pixels is None:
        bad_pixels = get_bad_pixels(
            calib_file=parameters_filename,
            dark_histo=dark_filename,
            plot=None
        )
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

    events = calibration_event_stream(files, max_events=max_events,
                                      disable_bar=disable_bar)
    events = baseline.fill_dark_baseline(events, dark_baseline)
    events = baseline.fill_digicam_baseline(events)
    events = tagging.tag_burst_from_moving_average_baseline(events)
    events = baseline.compute_baseline_shift(events)
    events = baseline.subtract_baseline(events)
    # events = baseline.compute_baseline_std(events, n_events=100)
    events = filters.filter_clocked_trigger(events)
    events = baseline.compute_nsb_rate(events, gain_amplitude,
                                       pulse_area, crosstalk,
                                       bias_resistance, cell_capacitance)
    events = baseline.compute_gain_drop(events, bias_resistance,
                                        cell_capacitance)
    events = peak.find_pulse_with_max(events)
    events = charge.compute_dynamic_charge(events,
                                           integral_width=integral_width,
                                           saturation_threshold=saturation_threshold,
                                           threshold_pulse=threshold_pulse,
                                           debug=debug,
                                           pulse_tail=False,)
    events = charge.compute_photo_electron(events, gains=gain)
    events = charge.interpolate_bad_pixels(events, geom, bad_pixels)

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
        data_to_store.burst = event.data.burst
        data_to_store.saturated = event.data.saturated

        for key, val in event.hillas.items():
            data_to_store[key] = val

        output_file.add_container(data_to_store)
    output_file.close()


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    max_events = convert_int(args['--max_events'])
    dark_filename = args['--dark']
    output = args['--output']
    output_path = os.path.dirname(output)
    if output_path != "" and not os.path.exists(output_path):
        raise IOError('Path ' + output_path +
                      'for output hillas does not exists \n')
    bad_pixels = convert_list_int(args['--bad_pixels'])
    integral_width = int(args['--integral_width'])
    picture_threshold = float(args['--picture_threshold'])
    boundary_threshold = float(args['--boundary_threshold'])
    debug = args['--debug']
    parameters_filename = args['--parameters']
    template_filename = args['--template']
    disable_bar = args['--disable_bar']
    saturation_threshold = float(args['--saturation_threshold'])
    threshold_pulse = float(args['--threshold_pulse'])
    main_pipeline(
        files=files,
        max_events=max_events,
        dark_filename=dark_filename,
        integral_width=integral_width,
        debug=debug,
        parameters_filename=parameters_filename,
        hillas_filename=output,
        picture_threshold=picture_threshold,
        boundary_threshold=boundary_threshold,
        template_filename=template_filename,
        bad_pixels=bad_pixels,
        disable_bar=disable_bar,
        threshold_pulse=threshold_pulse,
        saturation_threshold=saturation_threshold,
    )


if __name__ == '__main__':
    entry()
