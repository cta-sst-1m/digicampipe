import pkg_resources
import os

from digicampipe.processors.baseline import BaselineCalculatorFromWaveform
from digicampipe.io.zfits import zfits_event_source
import numpy as np

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)

digicam_config_file = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'camera_config.cfg'
    )
)


def test_baseline_computation_from_waveform():

    bins = np.random.randint(0, 50)

    process = BaselineCalculatorFromWaveform(bins=bins)
    computed_baseline = []
    expected_baseline = []

    for event in zfits_event_source(example_file_path):

        process(event)

        computed_baseline.append(event.r0.tel[1].baseline)
        temp = np.mean(event.r0.tel[1].adc_samples[..., bins], axis=0)
        expected_baseline.append(temp)

    computed_baseline = np.array(computed_baseline)
    expected_baseline = np.array(expected_baseline)

    assert (computed_baseline == expected_baseline).all()