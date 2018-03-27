from digicampipe.io.containers_calib import CalibrationHistogramContainer
import numpy as np
from histogram.histogram import Histogram1D


def convert_histogram_to_container(histogram):
    """
    Utility function to convert an Histogram to a CalibrationHistogramContainer
    :param histogram:
    :return: CalibrationHistogramContainer
    """

    container = CalibrationHistogramContainer()

    # TODO make ctapipe.HDFTableWriter accept unit32
    container.bins = histogram.bins.astype(np.int)
    container.count = histogram.data.astype(np.int)
    container.shape = histogram.shape  # TODO need to accept tuple
    container.n_bins = histogram.n_bins
    container.name = histogram.name  # TODO need to accept str
    container.axis_name = histogram.axis_name  # TODO need to accept str
    container.underflow = histogram.underflow.astype(np.int)
    container.overflow = histogram.overflow.astype(np.int)
    container.max = histogram.max
    container.min = histogram.min
    container.mode = histogram.mode()
    container.std = histogram.std()
    container.mean = histogram.mean()

    return container


def convert_container_to_histogram(container):

    if isinstance(container, CalibrationHistogramContainer):

        histo = Histogram1D(bin_edges=container.bins,
                            data_shape=container.count.shape,
                            name=container.name,
                            axis_name=container.axis_name)

        histo.data = container.count
        histo.underflow = container.underflow
        histo.overflow = container.overflow
        histo.max = container.max
        histo.min = container.min

        return histo

    else:

        raise TypeError
