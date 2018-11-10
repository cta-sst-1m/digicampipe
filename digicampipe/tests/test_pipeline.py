import os
import tempfile

from astropy.io import fits
from pkg_resources import resource_filename

from digicampipe.scripts.pipeline import main as main_pipeline
from digicampipe.scripts.raw import compute as compute_raw
from digicampipe.utils.docopt import convert_pixel_args

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
template_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)
expected_columns = ['phi', 'y', 'skewness', 'intensity', 'x', 'event_id',
                    'local_time', 'psi', 'width', 'miss', 'alpha', 'length',
                    'r', 'kurtosis', 'event_type']


def test_pipeline():
    # checks that the pipeline produce a fits file with all columns
    with tempfile.TemporaryDirectory() as tmpdirname:
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        hillas_filename = os.path.join(tmpdirname, 'hillas.fits')
        compute_raw(
            files=[example_file1_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename
        )
        main_pipeline(
            files=[example_file2_path],
            max_events=None,
            dark_filename=dark_filename,
            shift=0,
            integral_width=7,
            debug=False,
            hillas_filename=hillas_filename,
            template_filename=template_filename,
            parameters_filename=calibration_filename,
            compute=True,
            display=False,
            picture_threshold=1,  # unusual value, so events pass cuts
            boundary_threshold=1,  # unusual value, so events pass cuts
        )
        hdul = fits.open(os.path.join(tmpdirname, 'hillas.fits'))
        cols = [c.name for c in hdul[1].columns]
        nevent = len(hdul[1].data['local_time'])
        assert nevent > 0
        for col in expected_columns:
            assert col in cols
            assert len(hdul[1].data[col]) == nevent


def test_pipeline_two_pixels():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        hillas_filename = os.path.join(tmpdirname, 'hillas.fits')
        compute_raw(
            files=[example_file1_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename
        )
        main_pipeline(
            files=[example_file2_path],
            max_events=None,
            dark_filename=dark_filename,
            shift=0,
            integral_width=7,
            debug=False,
            hillas_filename=hillas_filename,
            template_filename=template_filename,
            parameters_filename=calibration_filename,
            compute=True,
            display=False,
            picture_threshold=1,  # unusual value, so events pass cuts
            boundary_threshold=1,  # unusual value, so events pass cuts
            bad_pixels=[0, 1],
        )
        hdul = fits.open(os.path.join(tmpdirname, 'hillas.fits'))
        cols = [c.name for c in hdul[1].columns]
        nevent = len(hdul[1].data['local_time'])
        assert nevent > 0
        for col in expected_columns:
            assert col in cols
            assert len(hdul[1].data[col]) == nevent


if __name__ == '__main__':
    test_pipeline_two_pixels()
