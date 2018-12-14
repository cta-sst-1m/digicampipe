import os
import tempfile
import numpy as np
from pkg_resources import resource_filename
from astropy import units as u

from digicampipe.scripts.nsb_rate_camera import nsb_rate
from digicampipe.scripts.raw import compute_baseline_histogram, compute


dark200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_200_evts.fits.fz'
    )
)
science200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'run150_200_evts.fits.fz'
    )
)
param_file = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)
template_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_template_all_pixels.txt'
    )
)


def test_nsb_rate():
    with tempfile.TemporaryDirectory() as tmpdirname:
        baseline_histo_file = os.path.join(tmpdirname, 'baseline.pk')
        compute_baseline_histogram(
            files=[science200_file_path],
            filename=baseline_histo_file
        )
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        compute(files=[dark200_file_path], filename=dark_filename)
        plot = os.path.join(tmpdirname, 'nsb.png')
        nsb_pixels = nsb_rate(baseline_histo_file, dark_filename, param_file,
                              template_filename, plot=plot)
        assert np.all(nsb_pixels > 0)
        assert np.all(nsb_pixels < 10 * u.GHz)
        assert os.path.isfile(plot)


if __name__ == '__main__':
    test_nsb_rate()
