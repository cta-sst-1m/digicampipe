import os

import pkg_resources

from digicampipe.scripts import rate_scan

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'digicamtoy',
        'test_digicamtoy_0.hdf5'
    )
)


def test_compute_rate_scan():
    rate_scan.compute(example_file_path, 'test')
