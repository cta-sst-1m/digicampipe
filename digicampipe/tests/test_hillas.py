import os
import tempfile
import numpy as np
from pkg_resources import resource_filename

from digicampipe.image.hillas import correct_alpha_2, \
    correct_alpha_3, correct_alpha_4
from digicampipe.utils.docopt import convert_pixel_args
from digicampipe.scripts.pipeline import main_pipeline
from digicampipe.scripts.plot_pipeline import get_data_and_selection
from digicampipe.scripts.raw import compute as compute_raw

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
science200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'run150_200_evts.fits.fz'
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
        'pulse_template_all_pixels.txt'
    )
)
aux_basepath = os.path.dirname(science200_file_path)


def get_data():
    with tempfile.TemporaryDirectory() as tmpdirname:
        hillas_filename = os.path.join(tmpdirname, 'hillas.fits')
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        compute_raw(
                files=[dark200_file_path],
                max_events=None,
                pixel_id=convert_pixel_args(None),
                filename=dark_filename,
                disable_bar=True
        )
        main_pipeline(
            files=[science200_file_path],
            aux_basepath=aux_basepath,
            max_events=None,
            dark_filename=dark_filename,
            integral_width=7,
            debug=False,
            hillas_filename=hillas_filename,
            template_filename=template_filename,
            parameters_filename=calibration_filename,
            picture_threshold=30,
            boundary_threshold=15,
            saturation_threshold=3000,
            threshold_pulse=0.1,
            disable_bar=True
        )
        data, _ = get_data_and_selection(hillas_filename)
    return data


def test_correct_alphas():
    data = get_data()
    data_alpha2 = correct_alpha_2(data, source_x=1, source_y=1)
    data_alpha3 = correct_alpha_3(data, source_x=1, source_y=1)
    alpha4 = correct_alpha_4(
        data, sources_x=[1, -1], sources_y=[1, -1]
    )
    assert np.all(data_alpha3 == data_alpha2)
    assert np.all(data_alpha3['alpha'] == alpha4[:, 0])


if __name__ == '__main__':
    test_correct_alphas()
