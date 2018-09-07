from pkg_resources import resource_filename
import os
import numpy as np

from digicampipe.utils.pulse_template import PulseTemplate

template_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)

PULSE_AREA = 17.974891497703858


def test_pulse_template_creation_with_file():

    template = PulseTemplate.load(template_filename)

    t, x = np.loadtxt(template_filename).T

    assert (t == template.time).all()


def test_pulse_template_integral():

    template = PulseTemplate.load(template_filename)

    assert template.integral() == PULSE_AREA


def test_pulse_template_plot():

    template = PulseTemplate.load(template_filename)

    template.plot()


def test_pulse_template_normalization():

    template = PulseTemplate.load(template_filename)

    assert np.max(template.amplitude) == 1

    t, x = np.loadtxt(template_filename).T
    # Trying with negative template
    template = PulseTemplate(-x, t)

    assert np.max(template.amplitude) == 1

    # Trying with non normalized template
    template = PulseTemplate(x*0.1, t)

    assert np.max(template.amplitude) == 1


def test_pulse_template_ndarray_amplitude():

    n_samples = 51
    time = np.linspace(0, 50, num=n_samples)
    amplitude = [np.ones(n_samples), np.ones(n_samples)]
    amplitude = np.array(amplitude)

    template = PulseTemplate(amplitude, time)
    y = template(time, amplitude=2)

    assert (template.integral() == np.array([50, 50])).all()
    np.testing.assert_almost_equal(y, amplitude * 2)


if __name__ == '__main__':

    test_pulse_template_plot()
    import matplotlib.pyplot as plt
    plt.show()
