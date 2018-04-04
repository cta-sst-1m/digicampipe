from histogram.histogram import Histogram1D
from digicampipe.io.containers_calib import CalibrationHistogramContainer
import numpy as np


def test_container_to_histogram():

    bins = np.arange(100)
    histo = Histogram1D(bin_edges=bins, data_shape=(1, ))
    container = CalibrationHistogramContainer()

    for i in range(10):

        data = np.arange(100).reshape(1, -1)
        histo.fill(data)

    container.from_histogram(histo)
    histo_from_container = container.to_histogram()

    for key, val in histo_from_container.__dict__.items():

        if isinstance(val, np.ndarray):

            assert (val == getattr(histo, key)).all(), '{} not equal'.\
                format(key)

        else:

            assert (val == getattr(histo, key)), '{} not equal'.format(key)


