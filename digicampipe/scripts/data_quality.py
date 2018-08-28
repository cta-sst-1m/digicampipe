"""
Make a quick data quality check
Usage:
  digicam-data-quality [options] [--] <INPUT>...

Options:
  --help            Show this
  --time_step=N     Time window in nanoseconds within which values are computed
                    [Default: 5000000000]
  --output=PATH     Output path
                    [Default: .]
  --compute         boolean, if true create data_quality.fits and baseline_histo.pk
  --display         boolean, if true read the output files of compute and
                    plot history of baseline and trigger rate.
"""
from docopt import docopt
import os
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


def entry(files, time_step, output_path, compute, display):
    pixel_id = np.arange(1296)
    n_pixels = len(pixel_id)

    filename = os.path.join(output_path, 'data_quality.fits')
    histo_filename = os.path.join(output_path, 'baseline_histo.pk')

    if not os.path.exists(output_path):
        raise IOError('Path {} for output '
                      'does not exists \n'.format(output_path))

    if compute:

        events = calibration_event_stream(files)

        time = 0
        baseline = 0
        count = 0

        container = DataQualityContainer()
        file = Serializer(filename, mode='w', format='fits')
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

            print(time)

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

    if display:

        data = Table.read(filename, format='fits')
        data = data.to_pandas()

        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')

        # baseline_histo = Histogram1D.load(histo_filename)
        # baseline_mean = baseline_histo.mean()
        # baseline_std = baseline_histo.std()

        # plt.figure()
        # plt.bar(pixel_id, baseline_mean)
        # plt.errorbar(pixel_id, baseline_mean, yerr=baseline_std)
        # plt.xlabel('pixel ')

        plt.figure()
        plt.plot(data['trigger_rate']*1E9)
        plt.ylabel('Trigger rate [Hz]')

        plt.figure()
        plt.plot(data['baseline'])
        plt.ylabel('Baseline [LSB]')

        plt.show()

    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    time_step = float(args['--time_step'])
    output_path = args['--output']
    compute = args['--compute']
    display = args['--display']
    entry(files, time_step, output_path, compute, display)
