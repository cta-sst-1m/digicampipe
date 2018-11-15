import os
import pkg_resources
import numpy as np

from digicampipe.scripts import rate_scan

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_200_evts.fits.fz'
    )
)


def test_compute_rate_scan():

    out = rate_scan.compute(example_file_path, 'test.fits',
                            thresholds=np.arange(0, 100, 1), n_samples=1024)

    # Check that maximum rate is 5 MHz

    assert out[0][0] == 5 * 1E-3
