import os
import tempfile

import numpy as np
from astropy.io import fits
from pkg_resources import resource_filename

from digicampipe.scripts.data_quality import main as data_quality
from digicampipe.scripts.raw import compute as compute_raw
from digicampipe.utils.docopt import convert_pixel_args

dark100_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_100_evts.fits.fz'
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
science100_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'run150_100_evts.fits.fz'
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
parameters_filename = resource_filename(
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
        'pulse_SST-1M_pixel_0.dat'
    )
)

expected_columns = ['time', 'baseline', 'trigger_rate', 'shower_rate',
                    'nsb_rate', 'burst']


def test_data_quality():
    files = [science200_file_path]
    time_step = 1e8  # in ns, average history plot over 100 ms
    with tempfile.TemporaryDirectory() as tmpdirname:
        fits_filename = os.path.join(tmpdirname, 'ouptput.fits')
        histo_filename = os.path.join(tmpdirname, 'ouptput.pk')
        rate_plot_filename = os.path.join(tmpdirname, 'rate.png')
        baseline_plot_filename = os.path.join(tmpdirname, 'baseline.png')
        nsb_plot_filename = os.path.join(tmpdirname, 'nsb.png')
        load_files = False
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        compute_raw(
            files=[dark100_file_path, dark200_file_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename,
            disable_bar=True
        )
        data_quality(
            files=files,
            dark_filename=dark_filename,
            time_step=time_step,
            fits_filename=fits_filename,
            load_files=load_files,
            histo_filename=histo_filename,
            rate_plot_filename=rate_plot_filename,
            baseline_plot_filename=baseline_plot_filename,
            nsb_plot_filename=nsb_plot_filename,
            parameters_filename=parameters_filename,
            template_filename=template_filename,
            threshold_sample_pe=20,
            disable_bar=True
        )
        hdul = fits.open(fits_filename)
        assert np.all(np.diff(hdul[1].data['time']) > 0)
        fits_columns = [c.name for c in hdul[1].columns]
        n_time = len(hdul[1].data[expected_columns[0]])
        assert n_time > 0

        assert len(fits_columns) == len(expected_columns)
        for col in expected_columns:
            assert col in fits_columns
            assert n_time == len(hdul[1].data[col])
        assert os.path.isfile(rate_plot_filename)
        assert os.path.isfile(baseline_plot_filename)


if __name__ == '__main__':
    test_data_quality()
