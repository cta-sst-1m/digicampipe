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

    return container





