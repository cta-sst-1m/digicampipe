from digicampipe.io.containers_calib import CalibrationHistogramContainer


def convert_histogram_to_container(histogram):
    """
    Utility function to convert an Histogram to a CalibrationHistogramContainer
    :param histogram:
    :return: CalibrationHistogramContainer
    """

    container = CalibrationHistogramContainer()

    container.bins = histogram.bins
    container.count = histogram.data
    container.shape = histogram.shape
    container.n_bins = histogram.n_bins
    container.name = histogram.name
    container.axis_name = histogram.axis_name
    container.underflow = histogram.underflow
    container.overflow = histogram.overflow
    container.max = histogram.max
    container.min = histogram.min
    container.mode = histogram.mode()
    container.std = histogram.std()
    container.mean = histogram.mean()

    return container





