import os
import tempfile
import numpy as np
import yaml
from pkg_resources import resource_filename

from digicampipe.scripts.bad_pixels import get_bad_pixels


parameters_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)

BAD_PIXELS = [415, 558, 682, 758, 790, 853, 876, 899, 952, 1007, 1049, 1113]


def test_bad_pixels():
    bad_pixels = get_bad_pixels(
        parameters_filename, nsigma_gain=5, nsigma_elecnoise=5,
        plot=None, output=None
    )
    assert np.all(bad_pixels == BAD_PIXELS)


def test_bad_pixels_plot():
    with tempfile.TemporaryDirectory() as tmpdirname:
        plot = os.path.join(tmpdirname, 'test.png')
        get_bad_pixels(parameters_filename, plot=plot, output=None)
        assert os.path.isfile(plot)


def test_bad_pixels_save():
    with tempfile.TemporaryDirectory() as tmpdirname:
        out = os.path.join(tmpdirname, 'test.png')
        get_bad_pixels(parameters_filename, plot=None, output=out)
        assert os.path.isfile(out)
        with open(out) as file:
            params = yaml.load(file)
            assert 'bad_pixels' in params.keys()


if __name__ == '__main__':
    test_bad_pixels()
    test_bad_pixels_plot()
    test_bad_pixels_save()
