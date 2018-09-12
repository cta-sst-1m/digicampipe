from pkg_resources import resource_filename
import os
import numpy as np

from digicampipe.utils import NormalizedPulseTemplate

template_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)

PULSE_AREA = 17.974891497703858
RATIO_CHARGE_AMPLITUDE = 0.24164813864342138


def test_pulse_template_creation_with_file():

    template = NormalizedPulseTemplate.load(template_filename)

    t, x = np.loadtxt(template_filename).T

    assert (t == template.time).all()


def test_pulse_template_integral():

    template = NormalizedPulseTemplate.load(template_filename)

    assert template.integral() == PULSE_AREA


def test_pulse_template_plot():

    template = NormalizedPulseTemplate.load(template_filename)

    template.plot()


def test_pulse_template_normalization():

    template = NormalizedPulseTemplate.load(template_filename)

    assert np.max(template.amplitude) == 1

    t, x = np.loadtxt(template_filename).T
    # Trying with negative template
    template = NormalizedPulseTemplate(-x, t)

    assert np.max(template.amplitude) == 1

    # Trying with non normalized template
    template = NormalizedPulseTemplate(x * 0.1, t)

    assert np.max(template.amplitude) == 1


def test_pulse_template_ndarray_amplitude():

    n_samples = 51
    time = np.linspace(0, 50, num=n_samples)
    amplitude = [np.ones(n_samples), np.ones(n_samples)]
    amplitude = np.array(amplitude)

    template = NormalizedPulseTemplate(amplitude, time)
    y = template(time, amplitude=2)

    assert (template.integral() == np.array([50, 50])).all()
    np.testing.assert_almost_equal(y, amplitude * 2)


def test_pulse_template_object_get_sub_template():

    n_samples = 51
    time = np.linspace(0, 50, num=n_samples)
    amplitude = [np.ones(n_samples), np.arange(n_samples)]
    amplitude = np.array(amplitude)

    template = NormalizedPulseTemplate(amplitude, time)

    assert template[0].integral() == 50
    np.testing.assert_almost_equal(template[1].integral(), 50 / 2)


def test_charge_amplitude_ratio():

    template = NormalizedPulseTemplate.load(template_filename)
    ratio = template.compute_charge_amplitude_ratio(7, 4)

    assert RATIO_CHARGE_AMPLITUDE == ratio


if __name__ == '__main__':

    test_pulse_template_plot()
    import matplotlib.pyplot as plt
    plt.show()
