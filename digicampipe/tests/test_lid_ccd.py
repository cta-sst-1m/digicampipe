from digicampipe.image.sky_image import *

from pkg_resources import resource_filename
from glob import glob

example_lid_CCD_image_file_paths = glob(
    resource_filename('digicampipe', 'tests/resources/stars_on_lid/*.fits')
)


def test_find_stars():
    # find stars in lid CCD images:
    crop_pixels1 = [  # (850, 50),
        # (550, 500),
        (350, 900),
        # (600, 1350),
        (850, 1800), ]
    crop_pixels2 = [  # (1100, 650),
        # (900, 1000),
        (770, 1550),
        # (1000, 2050),
        (1300, 2400), ]
    lidccd_obs = LidCCDObservation(
        example_lid_CCD_image_file_paths,
        crop_pixels1,
        crop_pixels2,
        scale_low_images_deg=8.,
        scale_high_images_deg=12.,
        guess_ra_dec=(83.2, 26.2),
        guess_radius=10
        )
    nsolved = 0
    for lidccd_image in lidccd_obs.lidccd_images:
        if lidccd_image.wcs is not None:
            nsolved += 1
    assert nsolved > 0


if __name__ == '__main__':
    test_find_stars()
