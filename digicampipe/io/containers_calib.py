from ctapipe.core import Container, Map
from ctapipe.core import Field
from numpy import ndarray


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
    count = Field(ndarray, 'count fs fe')
    shape = Field(ndarray, 'shape fsd ')
    n_bins = Field(ndarray, 'n_binsf f')
    name = Field(ndarray, 'namef fsf')
    axis_name = Field(ndarray, 'axis_namef sf')
    underflow = Field(ndarray, 'underflow fs')
    overflow = Field(ndarray, 'overflow sf')
    max = Field(ndarray, 'maxs f')
    min = Field(ndarray, 'min sf')
    mean = Field(ndarray, 'means f')
    std = Field(ndarray, 'stdsf')
    mode = Field(ndarray, 'modesf sd')


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
    n_pixels = Field(int, 'number of pixels')
    data = CalibrationEventContainer()
    histo = Field(Map(CalibrationHistogramContainer))
    result = CalibrationResultContainer()

