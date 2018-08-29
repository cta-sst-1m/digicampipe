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
    with tempfile.TemporaryDirectory() as tmpdirname:
        fits_filename = os.path.join(tmpdirname, 'ouptput.fits')
        histo_filename = os.path.join(tmpdirname, 'ouptput.pk')
        rate_plot_filename = os.path.join(tmpdirname, 'rate.png')
        baseline_plot_filename = os.path.join(tmpdirname, 'baseline.png')
        load_files = False
        data_quality(
            files, time_step, fits_filename, load_files,
            histo_filename, rate_plot_filename, baseline_plot_filename
        )
        hdul = fits.open(fits_filename)
        assert np.all(np.diff(hdul[1].data['time']) > 0)
        fits_columns = [c.name for c in hdul[1].columns]
        n_time = len(hdul[1].data[expected_columns[0]])
        assert n_time > 0
        for col in expected_columns:
            assert col in fits_columns
            assert n_time == len(hdul[1].data[col])
        assert os.path.isfile(rate_plot_filename)
        assert os.path.isfile(baseline_plot_filename)


if __name__ == '__main__':
    test_data_quality()
