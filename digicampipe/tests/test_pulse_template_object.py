import os
import numpy as np
from pkg_resources import resource_filename
import tempfile

from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.visualization.plot import plot_pulse_templates

template_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)
template_filename_2 = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_template_all_pixels.txt'
    )
)
data_filename1 = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'template_scan_dac_250.fits.gz'
    )
)
data_filename2 = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'template_scan_dac_400.fits.gz'
    )
)
data_filename3 = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'template_scan_dac_450.fits.gz'
    )
)

PULSE_AREA = 17.974891497703858
RATIO_CHARGE_AMPLITUDE = 0.24164813864342138


def test_pulse_template_creation_with_file():
    template = NormalizedPulseTemplate.load(template_filename)
    t, x = np.loadtxt(template_filename).T
    assert (t == template.time).all()


def test_pulse_template_creation_with_file_with_std():
    template = NormalizedPulseTemplate.load(template_filename_2)
    t, x, std = np.loadtxt(template_filename_2).T
    assert np.all(t == template.time)


def test_pulse_template_integral():
    template = NormalizedPulseTemplate.load(template_filename)
    assert template.integral() == PULSE_AREA


def test_pulse_template_plot():
    template = NormalizedPulseTemplate.load(template_filename)
    template.plot()
    template.plot_interpolation()
    plot_pulse_templates([template_filename], xscale='linear', yscale='linear')
    plot_pulse_templates([template_filename], xscale='log', yscale='log')


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


def test_pulse_template_from_datafile():
    template1 = NormalizedPulseTemplate.create_from_datafile(data_filename1)
    template2 = NormalizedPulseTemplate.create_from_datafile(data_filename2)
    template3 = NormalizedPulseTemplate.create_from_datafile(data_filename3)
    template_load = NormalizedPulseTemplate.load(template_filename_2)
    time = np.linspace(-10, 30, num=101)
    std = template_load.std(time)
    assert np.all(
        np.abs(template1(time) - template_load(time)) < 5 * std)
    assert np.all(
        np.abs(template2(time) - template_load(time)) < 5 * std)
    assert np.all(
        np.abs(template3(time) - template_load(time)) < 5 * std)


def test_pulse_template_from_datafiles():
    files = [data_filename1, data_filename2, data_filename3]
    template = NormalizedPulseTemplate.create_from_datafiles(files)
    template_load = NormalizedPulseTemplate.load(template_filename_2)
    time = np.linspace(-10, 30, num=101)
    std = template_load.std(time)
    assert np.all(
        np.abs(template(time) - template_load(time)) < 5 * std)


def test_pulse_template_save():
    template_load = NormalizedPulseTemplate.load(template_filename_2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        template_saved_filename = os.path.join(tmpdirname, 'test.txt')
        template_load.save(template_saved_filename)
        template_saved = NormalizedPulseTemplate.load(template_saved_filename)
        assert np.all(template_saved.time == template_load.time)
        assert np.all(template_saved.amplitude == template_load.amplitude)
        assert np.all(template_saved.amplitude_std ==
                      template_load.amplitude_std)


def test_save_and_load_fits():

    data = np.ones(100)
    time = np.arange(100)

    template_saved = NormalizedPulseTemplate(amplitude=data, time=time)

    with tempfile.TemporaryDirectory(suffix='.fits') as tmp_dir:
        filename = os.path.join(tmp_dir, 'test.fits')
        template_saved.save(filename)

        template_loaded = NormalizedPulseTemplate.load(filename)

        assert np.all(template_saved.time == template_loaded.time)
        assert np.all(template_saved.amplitude == template_loaded.amplitude)
        assert np.all(template_saved.amplitude_std ==
                      template_loaded.amplitude_std)


def test_normalization_for_array_template():

    data_1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 5, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 13, -4, 5, 1, 1, 1, 1, 1, 1]])
    data_2 = -np.array([[1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 5, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 13, -4, 5, 1, 1, 1, 1, 1, 1]])
    time = np.arange(data_1.shape[1])
    template_positive = NormalizedPulseTemplate(amplitude=data_1, time=time)
    template_negative = NormalizedPulseTemplate(amplitude=data_2, time=time)

    assert (template_positive.amplitude.max(axis=-1) == 1).all()
    assert (template_negative.amplitude.max(axis=-1) == 1).all()


if __name__ == '__main__':
    test_normalization_for_array_template()
