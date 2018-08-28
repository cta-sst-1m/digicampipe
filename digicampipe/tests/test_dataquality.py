import tempfile
import os
import numpy as np
from pkg_resources import resource_filename
from astropy.io import fits
from digicampipe.scripts.data_quality import entry as data_quality


example_file1_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)

expected_columns = ['time', 'baseline', 'trigger_rate']


def test_data_quality():
    files = [example_file1_path]
    time_step = 1e6  # in ns, average history plot over 1 ms
    with tempfile.TemporaryFile() as fits_file:
        with tempfile.TemporaryFile() as histo_file:
            fits_filename = str(fits_file.name)
            histo_filename = str(histo_file.name)
            data_quality(files, time_step, fits_filename, histo_filename,
                         compute=True, display=False)
            hdul = fits.open(fits_filename)
            assert np.all(np.diff(hdul[1].data['time']) > 0)
            fits_columns = [c.name for c in hdul[1].columns]
            n_time = len(hdul[1].data[expected_columns[0]])
            assert n_time > 0
            for col in expected_columns:
                assert col in fits_columns
                assert n_time == len(hdul[1].data[col])


if __name__ == '__main__':
    test_data_quality()
