import os
import tempfile
import numpy as np
import yaml
from pkg_resources import resource_filename

from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.scripts.raw import compute as compute_raw
from digicampipe.utils.docopt import convert_pixel_args


parameters_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)
dark200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_200_evts.fits.fz'
    )
)

BAD_PIXELS_PARM = [
    415, 558, 682, 758, 790, 853, 876, 899, 952, 1007, 1049, 1113
]

BAD_PIXELS_DARK = [
    17, 66, 130, 300, 400, 734, 799, 876, 1010, 1038, 1049, 1112, 1113, 1223
]


def test_bad_pixels_params():
    bad_pixels = get_bad_pixels(
        calib_file=parameters_filename, nsigma_gain=5, nsigma_elecnoise=5,
        plot=None, output=None
    )
    assert np.all(bad_pixels == BAD_PIXELS_PARM)


def test_bad_pixels_dark():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        compute_raw(
            files=[dark200_file_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename
        )
        bad_pixels = get_bad_pixels(
            dark_histo=dark_filename, nsigma_dark=8,
            plot=None, output=None
        )
        assert np.all(bad_pixels == BAD_PIXELS_DARK)


def test_bad_pixels_all():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        compute_raw(
            files=[dark200_file_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename
        )
        bad_pixels = get_bad_pixels(
            calib_file=parameters_filename, nsigma_gain=5, nsigma_elecnoise=5,
            dark_histo=dark_filename, nsigma_dark=8,
            plot=None, output=None
        )
        bad_pixels_true = BAD_PIXELS_PARM
        bad_pixels_true.extend(BAD_PIXELS_DARK)
        bad_pixels_true = np.unique(np.sort(bad_pixels_true))
        assert np.all(bad_pixels == bad_pixels_true)



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
    test_bad_pixels_all()
    test_bad_pixels_plot()
    test_bad_pixels_save()
