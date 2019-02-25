import os
import tempfile
import numpy as np
from pkg_resources import resource_filename
import pandas as pd

from digicampipe.image.hillas import compute_alpha, correct_hillas
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


def test_alpha_computation_for_aligned_showers():
    x = np.linspace(-100, 100, num=100)
    y = np.linspace(-100, 100, num=100)
    np.random.shuffle(y)
    data = {'x': x, 'y': y}
    data['r'] = np.sqrt(data['x'] ** 2 + data['y']**2)
    data['phi'] = np.arctan2(data['y'], data['x'])
    data['psi'] = data['phi']
    data = pd.DataFrame(data)
    alpha_1 = np.array(compute_alpha(data['phi'], data['psi']))
    assert (alpha_1 == 0).all()


def test_alpha_computation_for_missaligned_showers():
    thetas = np.linspace(0, np.pi/2, num=100)
    for theta in thetas:
        miss_alignement = theta
        x = np.linspace(-100, 100, num=100)
        y = np.linspace(-100, 100, num=100)
        np.random.shuffle(y)
        data = dict({'x': x, 'y': y})
        data['r'] = np.sqrt(data['x'] ** 2 + data['y']**2)
        data['phi'] = np.arctan2(data['y'], data['x'])
        data['psi'] = data['phi'] + miss_alignement
        alpha_1 = compute_alpha(data['phi'], data['psi'])
        assert (np.isfinite(alpha_1).all())
        np.testing.assert_almost_equal(alpha_1, miss_alignement)


def test_correct_hillas():
    x = np.linspace(0, 100, num=5)
    y = np.linspace(0, 100, num=5)
    np.random.shuffle(y)
    data = dict({'x': x, 'y': y})
    data['r'] = np.sqrt(x ** 2 + y**2)
    data['phi'] = np.arctan2(data['y'], data['x'])
    data['psi'] = np.ones(len(x)) * np.pi
    x_corr, y_corr, r_corr, phi_corr = correct_hillas(data['x'], data['y'])
    assert (x_corr == x).all()
    assert (y_corr == y).all()
    assert (r_corr == data['r']).all()
    assert (phi_corr == data['phi']).all()
    x_corr, y_corr, r_corr, phi_corr = correct_hillas(data['x'],
                                                      data['y'],
                                                      source_x=100,
                                                      source_y=100)
    assert (x_corr == x - 100).all()
    assert (y_corr == y - 100).all()
    assert (x_corr**2 + y_corr**2 == (x - 100)**2 + (y - 100)**2).all()


if __name__ == '__main__':
    test_correct_hillas()
    test_alpha_computation_for_aligned_showers()
    test_alpha_computation_for_missaligned_showers()
