from ctapipe.core import Container, Map
from ctapipe.core import Field
from numpy import ndarray
import numpy as np
from histogram.histogram import Histogram1D


class CalibrationEventContainer(Container):
    """
    description test
    """
    # Raw

    adc_samples = Field(ndarray, 'the raw data')
    digicam_baseline = Field(ndarray, 'the baseline computed by the camera')

    # Processed

    baseline = Field(ndarray, 'the reconstructed baseline')
    pulse_mask = Field(ndarray, 'mask of adc_samples. True if the adc sample'
                                'contains a pulse  else False')
    reconstructed_amplitude = Field(ndarray, 'array of the same shape as '
                                             'adc_samples giving the'
                                             ' reconstructed pulse amplitude'
                                             ' for each adc sample')
    reconstructed_charge = Field(ndarray, 'array of the same shape as '
                                          'adc_samples giving the '
                                          'reconstructed charge for each adc '
                                          'sample')
    reconstructed_number_of_pe = Field(ndarray, 'estimated number of photon '
                                                'electrons for each adc sample'
                                       )


class CalibrationHistogramContainer(Container):
    """
    description test
    """

    bins = Field(ndarray, 'bins')
    count = Field(ndarray, 'count')
    shape = Field(ndarray, 'shape')
    n_bins = Field(ndarray, 'n_bins')
    name = Field(ndarray, 'name')
    axis_name = Field(ndarray, 'axis_name')
    underflow = Field(ndarray, 'underflow')
    overflow = Field(ndarray, 'overflow')
    max = Field(ndarray, 'max')
    min = Field(ndarray, 'min')
    mean = Field(ndarray, 'mean')
    std = Field(ndarray, 'std')
    mode = Field(ndarray, 'mode')

    def from_histogram(self, histogram):
        """
        Utility function to convert an Histogram to a
        CalibrationHistogramContainer
        :param histogram:
        :return: CalibrationHistogramContainer
        """

        # TODO make ctapipe.HDFTableWriter accept unit32, tuple, str
        self.bins = histogram.bins.astype(np.int)
        self.count = histogram.data.astype(np.int)
        self.shape = histogram.shape[:-1]  # TODO need to accept tuple
        self.n_bins = histogram.n_bins
        self.name = histogram.name  # TODO need to accept str
        self.axis_name = histogram.axis_name  # TODO need to accept str
        self.underflow = histogram.underflow.astype(np.int)
        self.overflow = histogram.overflow.astype(np.int)
        self.max = histogram.max
        self.min = histogram.min
        self.mode = histogram.mode()
        self.std = histogram.std()
        self.mean = histogram.mean()

        return self

    def to_histogram(self):

        histo = Histogram1D(bin_edges=self.bins,
                            data_shape=self.count.shape[:-1],
                            name=self.name,
                            axis_name=self.axis_name)

        histo.data = self.count
        histo.underflow = self.underflow
        histo.overflow = self.overflow
        histo.max = self.max
        histo.min = self.min

        return histo


class CalibrationResultContainer(Container):

    pass


class SPEParameters(Container):

    a_1 = Field(ndarray, 'Amplitude of the 1 p.e. peak')
    a_2 = Field(ndarray, 'Amplitude of the 2 p.e. peak')
    a_3 = Field(ndarray, 'Amplitude of the 3 p.e. peak')
    a_4 = Field(ndarray, 'Amplitude of the 4 p.e. peak')
    baseline = Field(ndarray, 'Position of the 0 p.e. peak')
    gain = Field(ndarray, 'Gain')
    sigma_e = Field(ndarray, 'Electronic noise')
    sigma_s = Field(ndarray, 'Sensor noise')
    dark_count = Field(ndarray, 'Dark count rate')
    crosstalk = Field(ndarray, 'Crosstalk')
    pixel = Field(ndarray, 'pixel id')


class SPEResultContainer(CalibrationResultContainer):
    """
    Container holding the results of the Single Photo Electron Spectrum
    analysis
    """

    init = Field(SPEParameters())
    bound_min = Field(SPEParameters())
    bound_max = Field(SPEParameters())
    param = Field(SPEParameters())
    param_errors = Field(SPEParameters())


class CalibrationContainer(Container):
    """
    This Container() is used for the camera calibration pipeline.
    It is meant to save each step of the calibration pipeline
    """

    config = Field(list, 'List of the input parameters'
                         ' of the calibration analysis')  # Should use dict?
    pixel_id = Field(ndarray, 'pixel ids')
    data = CalibrationEventContainer()
    histo = Field(Map(CalibrationHistogramContainer))
    result = CalibrationResultContainer()
