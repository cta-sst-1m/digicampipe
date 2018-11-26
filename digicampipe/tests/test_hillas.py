import os
import tempfile
import numpy as np
from pkg_resources import resource_filename
import astropy.units as u
from astropy.table import Table
import pandas as pd

from digicampipe.image.hillas import correct_alpha_2, \
    correct_alpha_3, correct_alpha_4, compute_alpha, correct_hillas
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
    data =get_data()
    data_alpha1 =  Table.from_pandas(
        correct_hillas(data, source_x=1, source_y=1)
    )
    data_alpha2 = correct_alpha_2(data, source_x=1, source_y=1)
    data_alpha3 = correct_alpha_3(data, source_x=1, source_y=1)
    alpha4 = correct_alpha_4(
        data, sources_x=[1, -1], sources_y=[1, -1]
    )
    data_alpha1['phi'] = data_alpha1['phi'] * u.rad
    data_alpha1['psi'] = data_alpha1['psi'] * u.rad
    alpha1 = compute_alpha(data_alpha1)
    assert np.all(alpha1.value == np.array(data_alpha2['alpha']))
    assert np.all(data_alpha3 == data_alpha2)
    assert np.all(data_alpha3['alpha'] == alpha4[:, 0])


def test_alpha_computation_for_aligned_showers():

    x = np.linspace(-100, 100, num=100)
    y = np.linspace(-100, 100, num=100)
    np.random.shuffle(y)
    data = {'x': x, 'y': y}
    data['r'] = np.sqrt(data['x'] ** 2 + data['y']**2)
    data['phi'] = np.arctan2(data['y'], data['x'])
    data['psi'] = data['phi']

    data = pd.DataFrame(data)

    alpha_1 = np.array(compute_alpha(data))
    alpha_2 = np.array(correct_alpha_2(data)['alpha'])
    alpha_3 = np.array(correct_alpha_3(data)['alpha'])

    assert (alpha_1 == 0).all()
    assert (alpha_2 == 0).all()
    assert (alpha_3 == 0).all()


def test_alpha_computation_for_missaligned_showers():

    thetas = np.linspace(0.01, np.pi/2, num=100)

    for theta in thetas:

        miss_alignement = (np.pi / 2 - theta)

        x = np.linspace(-100, 100, num=100)
        y = np.linspace(-100, 100, num=100)
        np.random.shuffle(y)
        data = {'x': x, 'y': y}
        data['r'] = np.sqrt(data['x'] ** 2 + data['y']**2)
        data['phi'] = np.arctan2(data['y'], data['x'])
        data['psi'] = data['phi'] + miss_alignement
        miss_alignement = np.rad2deg(miss_alignement)

        data = pd.DataFrame(data)

        alpha_1 = np.array(compute_alpha(data))
        alpha_2 = np.array(correct_alpha_2(data)['alpha'])
        alpha_3 = np.array(correct_alpha_3(data)['alpha'])

        assert (np.isfinite(alpha_1).all())

        np.testing.assert_almost_equal(alpha_1, miss_alignement)
        # np.testing.assert_almost_equal(alpha_2, miss_alignement)
        # np.testing.assert_almost_equal(alpha_3, miss_alignement)


if __name__ == '__main__':
    test_alpha_computation_for_aligned_showers()
    test_correct_alphas()
