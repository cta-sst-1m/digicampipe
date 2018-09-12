from glob import glob

import pytest
from pkg_resources import resource_filename

example_lid_CCD_image_file_paths = glob(
    resource_filename('digicampipe', 'tests/resources/stars_on_lid/*.fits')
)


@pytest.mark.deselect
def test_find_stars(method='nova'):
    from digicampipe.image.lidccd.sky_image import LidCCDObservation
    from digicampipe.image.lidccd.utils import Rectangle
    # find stars in lid CCD images:
    rectangles = [
        Rectangle(350, 900, 770, 1550),
        Rectangle(850, 1800, 1300, 2400)
    ]
    lidccd_obs = LidCCDObservation(
        example_lid_CCD_image_file_paths,
        rectangles=rectangles,
        scale_low_images_deg=8.,
        scale_high_images_deg=12.,
        guess_ra_dec=(83.2, 26.2),
        guess_radius=10,
        method=method
    )
    nsolved = 0
    for lidccd_image in lidccd_obs.lidccd_images:
        if lidccd_image.wcs is not None:
            nsolved += 1
    assert nsolved > 0
