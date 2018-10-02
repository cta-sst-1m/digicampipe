"""
Make a quick data quality check
Usage:
  digicam-data-quality [options] [--] <INPUT>...

Options:
  --help                        Show this
  --dark_filename=FILE          path to histogram of the dark files
  --parameters=FILE             Calibration parameters file path
  --time_step=N                 Time window in nanoseconds within which values
                                are computed
                                [Default: 5000000000]
  --output-fits=FILE            path to output fits file
                                [Default: ./data_quality.fits]
  --output-hist=FILE            path to output histo file
                                [Default: ./baseline_histo.pk]
  --load                        If not present, the INPUT zfits files will be
                                analyzed and output fits and histo files will
                                be created. If present, that analysis is
                                skipped and the fits and histo files will serve
                                as input for plotting.
                                [Default: False]
  --rate_plot=FILE              path to the output plot history of rate.
                                Use "none" to not create the plot and "show" to
                                open an interactive plot instead of creating a
                                file.
                                [Default: none]
  --baseline_plot=FILE          path to the output plot history of the mean
                                baseline. Use "none" to not create the plot and
                                "show" to open an interactive plot instead of
                                creating a file.
                                [Default: none]
  --template=FILE               Pulse template file path
"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from ctapipe.core import Field
from ctapipe.io.containers import Container
from ctapipe.io.serializer import Serializer
from docopt import docopt
from histogram.histogram import Histogram1D
from numpy import ndarray

from digicampipe.calib.baseline import fill_digicam_baseline, \
    subtract_baseline, compute_gain_drop, compute_nsb_rate, \
    compute_baseline_shift, fill_dark_baseline, tag_burst
from digicampipe.calib.charge import compute_sample_photo_electron
from digicampipe.calib.cleaning import compute_3d_cleaning
from digicampipe.instrument.camera import DigiCam
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
import os
from matplotlib.dates import DateFormatter


class DataQualityContainer(Container):
    time = Field(ndarray, 'time')
    baseline = Field(ndarray, 'baseline average over the camera')
    trigger_rate = Field(ndarray, 'Digicam trigger rate')
    shower_rate = Field(ndarray, 'shower rate')
    burst = Field(bool, 'is there a burst')


def main(files, dark_filename, time_step, fits_filename, load_files,
          histo_filename, rate_plot_filename, baseline_plot_filename,
          parameters_filename, template_filename, bias_resistance=1e4 * u.Ohm,
          cell_capacitance=5e-14 * u.Farad):
    with open(parameters_filename) as file:
        calibration_parameters = yaml.load(file)

    pulse_template = NormalizedPulseTemplate.load(template_filename)
    pulse_area = pulse_template.integral() * u.ns
    gain_integral = np.array(calibration_parameters['gain'])

    charge_to_amplitude = pulse_template.compute_charge_amplitude_ratio(7, 4)
    gain_amplitude = gain_integral * charge_to_amplitude
    crosstalk = np.array(calibration_parameters['mu_xt'])
    pixel_id = np.arange(1296)
    n_pixels = len(pixel_id)
    dark_histo = Histogram1D.load(dark_filename)
    dark_baseline = dark_histo.mean()
    if not load_files:
        events = calibration_event_stream(files)
        events = fill_digicam_baseline(events)
        events = fill_dark_baseline(events, dark_baseline)
        events = subtract_baseline(events)
        events = compute_baseline_shift(events)
        events = compute_nsb_rate(
            events, gain_amplitude, pulse_area, crosstalk, bias_resistance,
            cell_capacitance
        )
        events = compute_gain_drop(events, bias_resistance, cell_capacitance)
        events = compute_sample_photo_electron(events, gain_amplitude)
        events = tag_burst(events, event_average=100, threshold_lsb=5)
        events = compute_3d_cleaning(events, geom=DigiCam.geometry, threshold_sample_pe=20)
        init_time = 0
        baseline = 0
        count = 0
        shower_count = 0
        container = DataQualityContainer()
        file = Serializer(fits_filename, mode='w', format='fits')
        baseline_histo = Histogram1D(
            data_shape=(n_pixels,),
            bin_edges=np.arange(4096)
        )
        burst = False
        for i, event in enumerate(events):
            new_time = event.data.local_time
            if init_time == 0:
                init_time = new_time
            count += 1
            baseline += np.mean(event.data.digicam_baseline)
            time_diff = new_time - init_time
            if event.data.shower:
                shower_count += 1
            if event.data.burst:
                burst = True
            baseline_histo.fill(event.data.digicam_baseline.reshape(-1, 1))
            if time_diff > time_step and i > 0:
                trigger_rate = count / time_diff
                shower_rate = shower_count / time_diff
                baseline = baseline / count
                container.trigger_rate = trigger_rate
                container.baseline = baseline
                container.time = (new_time + init_time) / 2
                container.shower_rate = shower_rate
                container.burst = burst
                baseline = 0
                count = 0
                init_time = 0
                shower_count = 0
                burst = False
                file.add_container(container)
        output_path = os.path.dirname(histo_filename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        baseline_histo.save(histo_filename)
        print(histo_filename, 'created.')
        file.close()
        print(fits_filename, 'created.')

    data = Table.read(fits_filename, format='fits')
    data = data.to_pandas()
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    if rate_plot_filename != "none":
        fig1 = plt.figure()
        ax = plt.gca()
        plt.xticks(rotation=70)
        plt.plot(data['trigger_rate']*1E9, '.', label='trigger rate')
        plt.plot(data['shower_rate']*1E9, '.', label='shower_rate')
        plt.ylabel('rate [Hz]')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.legend()
        if rate_plot_filename == "show":
            plt.show()
        else:
            output_path = os.path.dirname(rate_plot_filename)
            if not (output_path == '' or os.path.exists(output_path)):
                os.makedirs(output_path)
            plt.savefig(rate_plot_filename)
        plt.close(fig1)

    if baseline_plot_filename != "none":
        fig2 = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        data_burst = data[data['burst'] == True]
        data_good = data[data['burst'] == False]
        plt.xticks(rotation=70)
        plt.plot(data_good['baseline'], '.', label='good', ms=2)
        plt.plot(data_burst['baseline'], '.', label='burst', ms=2)
        plt.ylabel('Baseline [LSB]')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.legend()
        plt.tight_layout()
        if rate_plot_filename == "show":
            plt.show()
        else:
            output_path = os.path.dirname(baseline_plot_filename)
            if not (output_path == '' or os.path.exists(output_path)):
                os.makedirs(output_path)
            plt.savefig(baseline_plot_filename)
        plt.close(fig2)
    return


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    dark_filename = args['--dark_filename']
    time_step = float(args['--time_step'])
    fits_filename = args['--output-fits']
    histo_filename = args['--output-hist']
    load_files = args['--load']
    rate_plot_filename = args['--rate_plot']
    baseline_plot_filename = args['--baseline_plot']
    parameters_filename = args['--parameters']
    template_filename = args['--template']

    main(files, dark_filename, time_step, fits_filename, load_files,
          histo_filename, rate_plot_filename, baseline_plot_filename,
          parameters_filename, template_filename)

if __name__ == '__main__':
    entry()
