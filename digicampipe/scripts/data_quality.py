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
  --load                        If not present, the INPUT files will be
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
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy import ndarray
from ctapipe.io.containers import Container
from ctapipe.io.serializer import Serializer
from ctapipe.core import Field
from astropy.table import Table
from histogram.histogram import Histogram1D
from digicampipe.utils import DigiCam
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, \
    subtract_baseline, compute_gain_drop, compute_nsb_rate, \
    compute_baseline_shift, fill_dark_baseline
from digicampipe.calib.camera.charge import compute_fractional_photo_electron
from digicampipe.calib.camera.cleaning import compute_3d_cleaning

class DataQualityContainer(Container):

    time = Field(ndarray, 'time')
    baseline = Field(ndarray, 'baseline average over the camera')
    trigger_rate = Field(ndarray, 'Digicam trigger rate')
    shower_rate = Field(ndarray, 'shower rate')


def entry(files, dark_filename, time_step, fits_filename, load_files, histo_filename,
          rate_plot_filename, baseline_plot_filename, parameters_filename,
          bias_resistance=1e4, cell_capacitance = 5e-14, pulse_area=4):
    with open(parameters_filename) as file:
        calibration_parameters = yaml.load(file)
    gain_integral = np.array(calibration_parameters['gain'])
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
            events, gain_integral, pulse_area, crosstalk, bias_resistance,
            cell_capacitance
        )
        events = compute_gain_drop(events, bias_resistance, cell_capacitance)
        events = compute_fractional_photo_electron(events, gain_integral)
        events = compute_3d_cleaning(events, geom=DigiCam.geometry)
        init_time = 0
        baseline = 0
        count = 0
        shower_count = 0
        container = DataQualityContainer()
        file = Serializer(fits_filename, mode='w', format='fits')
        baseline_histo = Histogram1D(
            data_shape=(n_pixels, ),
            bin_edges=np.arange(4096)
        )
        for i, event in enumerate(events):
            new_time = event.data.local_time
            if init_time == 0:
                init_time = new_time
            count += 1
            baseline += np.mean(event.data.digicam_baseline)
            time_diff = new_time - init_time
            if event.data.shower:
                shower_count += 1
            baseline_histo.fill(event.data.digicam_baseline.reshape(-1, 1))
            if time_diff > time_step and i > 0:
                trigger_rate = count / time_diff
                shower_rate = shower_count / time_diff
                baseline = baseline / count
                container.trigger_rate = trigger_rate
                container.baseline = baseline
                container.time = init_time
                container.shower_rate = shower_rate
                baseline = 0
                count = 0
                init_time = 0
                shower_count = 0
                file.add_container(container)
        baseline_histo.save(histo_filename)
        file.close()

    data = Table.read(fits_filename, format='fits')
    data = data.to_pandas()
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    if rate_plot_filename != "none":
        fig1 = plt.figure()
        plt.plot(data['trigger_rate']*1E9)
        plt.plot(data['shower_rate']*1E9)
        plt.ylabel('rate [Hz]')
        plt.legend('trigger rate', 'shower_rate')
        if rate_plot_filename == "show":
            plt.show()
        else:
            plt.savefig(rate_plot_filename)
        plt.close(fig1)

    if baseline_plot_filename != "none":
        fig2 = plt.figure()
        plt.plot(data['baseline'])
        plt.ylabel('Baseline [LSB]')
        if rate_plot_filename == "show":
            plt.show()
        else:
            plt.savefig(baseline_plot_filename)
        plt.close(fig2)

    return


if __name__ == '__main__':
    import sys
    print(sys.argv)
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
    entry(files, dark_filename, time_step, fits_filename, load_files, histo_filename,
          rate_plot_filename, baseline_plot_filename, parameters_filename)
