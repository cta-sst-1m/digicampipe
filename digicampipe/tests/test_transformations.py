import numpy as np
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy.time import Time
from astropy import units as u

from digicampipe.utils.transformations import transform_azel_to_xy, \
    get_stars_in_fov, ifj_krakow


def test_transform_azel_to_xy():
    stars_x, stars_y = transform_azel_to_xy(
        stars_az=np.array([[0, 0], [10, 10]]) * u.deg,
        stars_alt=np.array([[0, 0], [10, 10]]) * u.deg,
        az_obs=np.array([0, 10]) * u.deg, el_obs=np.array([0, 10]) * u.deg
    )
    eps = 1e-6 * u.m
    assert np.abs(stars_x[0, 0]) < eps  # 1st star in center of fov for 1st obs
    assert np.abs(stars_y[0, 0]) < eps  # 1st star in center of fov for 1st obs
    assert np.abs(stars_x[1, 1]) < eps  # 2nd star in center of fov for 2nd obs
    assert np.abs(stars_y[1, 1]) < eps  # 2nd star in center of fov for 2nd obs
    assert np.abs(stars_x[0, 1]) > eps
    assert np.abs(stars_y[0, 1]) > eps
    assert np.abs(stars_x[1, 0]) > eps
    assert np.abs(stars_y[1, 0]) > eps


def test_get_stars_in_fov():
    time_obs = Time([1.5e8], format='unix')
    deneb = SkyCoord.from_name('Deneb').transform_to(
        AltAz(obstime=time_obs[0], location=ifj_krakow)
    )
    stars_az, stars_alt, stars_pmag = get_stars_in_fov(
        deneb.az, deneb.alt, time_obs, radius = Angle(0.1, "deg"),
        catalog = 'I/254', Pmag_max=2
    )
    assert len(stars_az) == 1
    assert stars_az[0, 0] - deneb.az < 1e-3 * u.deg
    assert stars_alt[0, 0] - deneb.alt < 1e-3 * u.deg


if __name__ == '__main__':
    test_transform_azel_to_xy()
    test_get_stars_in_fov()
