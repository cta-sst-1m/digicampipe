import numpy as np

from digicampipe.io.containers import CalibrationContainer
from digicampipe.calib.charge import compute_dynamic_charge
from digicampipe.calib.peak import find_pulse_with_max


def _make_dummy_stream():
    event = CalibrationContainer()

    n_pixels = 4
    n_samples = 100

    for i in range(80):
        event.data.adc_samples = np.zeros((n_pixels, n_samples))
        event.data.adc_samples[1][i:i+20] = 2000
        event.data.adc_samples[2][i:i+20] = 4000
        event.data.adc_samples[3] = event.data.adc_samples[2]
        event.data.adc_samples[3][i+21:] = -4000

        yield event


def test_dynamic_charge():

    events = _make_dummy_stream()

    events = find_pulse_with_max(events)

    integral_width = 5

    events = compute_dynamic_charge(events, integral_width=integral_width,
                                    threshold_pulse=0.1,
                                    saturation_threshold=3000)

    for event in events:

        charge = event.data.reconstructed_charge

        assert charge[0] == 0
        assert charge[1] == 2000 * (integral_width//2 + 1)
        assert charge[2] == 4000 * 20
        assert charge[3] == 4000 * 20

        assert event.data.saturated


