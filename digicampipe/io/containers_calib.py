from ctapipe.core import Container, Map
from ctapipe.core import Field
from numpy import ndarray


class CalibrationEventContainer(Container):
    # Raw

    adc_samples = Field(ndarray, 'the raw data')
    digicam_baseline = Field(ndarray, 'the baseline computed by the camera')

    # Processed

    reconstructed_baseline = Field(ndarray, 'the reconstructed baseline')
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

    bin_edges = Field(ndarray, '')
    count = Field(ndarray, '')


class CalibrationResultContainer(Container):

    pass


class SPEParameters(Container):

    a_1 = Field(ndarray, 'Amplitude of the 1 p.e. peak')
    a_2 = Field(ndarray, '')
    a_3 = Field(ndarray, '')
    a_4 = Field(ndarray, '')
    baseline = Field(ndarray, 'Position of the 0 p.e. peak')
    gain = Field(ndarray, 'Gain')
    sigma_e = Field(ndarray, 'Electronic noise')
    sigma_s = Field(ndarray, 'Sensor noise')
    dark_count = Field(ndarray, 'Dark count rate')
    crosstalk = Field(ndarray, 'Crosstalk')


class SPEResultContainer(CalibrationResultContainer):
    """
    Container holding the results of the Single Photo Electron Spectrum
    analysis
    """

    init = SPEParameters()
    bound_min = SPEParameters()
    bound_max = SPEParameters()
    param = SPEParameters()
    param_errors = SPEParameters()


class CalibrationContainer(Container):
    """
    This Container() is used for the camera calibration pipeline.
    It is meant to save each step of the calibration pipeline
    """

    config = Field(list, 'List of the input parameters'
                         ' of the calibration analysis')  # Should use dict?
    n_pixels = Field(int, 'number of pixels')
    data = CalibrationEventContainer(None, 'Contains the raw data as well as the'
                                          'intermidiate steps'
                                          ' of the p.e. reconstruction')
    histo = CalibrationHistogramContainer(list, 'A list of the histograms'
                                                ' of the data')
    result = CalibrationResultContainer()

