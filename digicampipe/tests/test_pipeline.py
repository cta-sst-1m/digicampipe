import tempfile
import os
import numpy as np
from pkg_resources import resource_filename
from astropy.io import fits
from digicampipe.scripts.pipeline import main as main_pipeline
from digicampipe.scripts.raw import compute as compute_raw
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args

example_file1_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)
example_file2_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_10_evts.000.fits.fz'
    )
)
calibration_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)


def test_pipeline():
    # checks that the pipeline produce a fits file with all columns
    with tempfile.TemporaryFile() as dark_file:
        dark_filename = str(dark_file.name)
        compute_raw(
            files=[example_file1_path], 
            max_events=None, 
            pixel_id=convert_pixel_args(None), 
            filename=dark_filename
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            main_pipeline(
                files=[example_file2_path],
                max_events=None, 
                dark_filename=dark_filename, 
                pixel_ids=convert_pixel_args(None), 
                shift=0, 
                integral_width=7,
                debug=True, 
                output_path=tmpdirname, 
                parameters_filename=calibration_filename, 
                compute=True, 
                display=False,
                picture_threshold=1, # unusual value, so events pass cuts
                boundary_threshold=1 # unusual value, so events pass cuts
            )
            hdul = fits.open(os.path.join(tmpdirname, 'hillas.fits'))
            cols = [c.name for c in hdul[1].columns]
            assert 'phi' in cols
            assert 'y' in cols
            assert 'skewness' in cols
            assert 'intensity' in cols
            assert 'x' in cols
            assert 'event_id' in cols
            assert 'local_time' in cols
            assert 'psi' in cols
            assert 'width' in cols
            assert 'miss' in cols
            assert 'alpha' in cols
            assert 'length' in cols
            assert 'r' in cols
            assert 'kurtosis' in cols
            assert 'event_type' in cols


if __name__ == '__main__':
    test_pipeline()
