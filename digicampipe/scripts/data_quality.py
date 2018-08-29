"""
Make a quick data quality check
Usage:
  digicam-data-quality [options] [--] <INPUT>...

Options:
  --help            Show this
  --time_step=N     Time window in nanoseconds within which values are computed
                    [Default: 5000000000]
  --output-fits=PATH    path to output fits file
                    [Default: ./data_quality.fits]
  --output-hist=PATH    path to output histo file
                    [Default: ./baseline_histo.pk]
  --load            If not present, the INPUT files will be analyzed and
                    output fits and histo files will be created. If present,
                    that analysis is skipped and the fits and histo files will
                    serve as input for plotting.
                    [Default: False]
  --rate_plot=PATH  path to the output plot history of rate.
                    Use "none" to not create the plot and "show" to open an
                    interactive plot instead of creating a file.
                    [Default: none]
  --baseline_plot_filename=PATH path to the output plot history of the mean
                    baseline.
                    Use "none" to not create the plot and "show" to open an
                    interactive plot instead of creating a file.
                    [Default: none]
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from ctapipe.io.containers import Container
from ctapipe.io.serializer import Serializer
from ctapipe.core import Field
from astropy.table import Table
from histogram.histogram import Histogram1D

from digicampipe.io.event_stream import calibration_event_stream


class DataQualityContainer(Container):

    time = Field(ndarray, 'time')
    baseline = Field(ndarray, 'baseline average over the camera')
    trigger_rate = Field(ndarray, 'Digicam trigger rate')


def entry(files, time_step, fits_filename, load_files, histo_filename,
          rate_plot_filename, baseline_plot_filename):
    pixel_id = np.arange(1296)
    n_pixels = len(pixel_id)
    if not load_files:
        events = calibration_event_stream(files)
        time = 0
        baseline = 0
        count = 0
        container = DataQualityContainer()
        file = Serializer(fits_filename, mode='w', format='fits')
        baseline_histo = Histogram1D(data_shape=(n_pixels, ),
                                     bin_edges=np.arange(4096))
        n_event = 0
        n_container = 0
        for i, event in enumerate(events):
            n_event += 1
            new_time = event.data.local_time
            count += 1
            baseline += np.mean(event.data.digicam_baseline)
            time_diff = new_time - time
            time = new_time
            baseline_histo.fill(event.data.digicam_baseline.reshape(-1, 1))
            if time_diff > time_step and i > 0:
                trigger_rate = count / time_diff
                baseline = baseline / count
                container.trigger_rate = trigger_rate
                container.baseline = baseline
                container.time = time
                baseline = 0
                count = 0
                n_container += 1
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
        plt.ylabel('Trigger rate [Hz]')
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
    args = docopt(__doc__)
    files = args['<INPUT>']
    time_step = float(args['--time_step'])
    fits_filename = args['--output-fits']
    histo_filename = args['--output-hist']
    load_files = args['--load']
    rate_plot_filename = args['--rate_plot']
    baseline_plot_filename = args['--baseline_plot']
    entry(files, time_step, fits_filename, load_files, histo_filename,
          rate_plot_filename, baseline_plot_filename)
