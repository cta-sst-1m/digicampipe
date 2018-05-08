import pkg_resources
import os

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


def test_main_rate_scan():

    rate_scan.main(example_file_path, '/null')
