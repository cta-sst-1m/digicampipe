from ctapipe.core import Container, Map
from ctapipe.core import Field
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D


class CalibrationEventContainer(Container):
    """
    description test
    """
    # Raw

    adc_samples = Field(ndarray, 'the raw data')
    digicam_baseline = Field(ndarray, 'the baseline computed by the camera')
    local_time = Field(ndarray, 'timestamps')

    # Processed

    dark_baseline = Field(ndarray, 'the baseline computed in dark')
    baseline_shift = Field(ndarray, 'the baseline shift')
    nsb_rate = Field(ndarray, 'Night sky background rate')
    gain_drop = Field(ndarray, 'Gain drop')
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

    reconstructed_time = Field(ndarray, 'reconstructed time '
                                        'for each adc sample')

    def plot(self, pixel_id):

        plt.figure()
        plt.title('pixel : {}'.format(pixel_id))
        plt.plot(self.adc_samples[pixel_id], label='raw')
        plt.plot(self.pulse_mask[pixel_id], label='peak position')
        plt.plot(self.reconstructed_charge[pixel_id], label='charge',
                 linestyle='None', marker='o')
        plt.plot(self.reconstructed_amplitude[pixel_id], label='amplitude',
                 linestyle='None', marker='o')
        plt.legend()


class CalibrationContainer(Container):
    """
    This Container() is used for the camera calibration pipeline.
    It is meant to save each step of the calibration pipeline
    """

    config = Field(list, 'List of the input parameters'
                         ' of the calibration analysis')  # Should use dict?
    pixel_id = Field(ndarray, 'pixel ids')
    data = CalibrationEventContainer()
