from pkg_resources import resource_filename
from glob import glob


example_lid_CCD_image_file_paths = glob(
    resource_filename('digicampipe', 'tests/resources/stars_on_lid/*.fits')
)


def test_find_stars(method='local'):
    from digicampipe.image.sky_image import LidCCDObservation
    from digicampipe.image.utils import Rectangle
    # find stars in lid CCD images:
    rectangles = [Rectangle(350, 900, 770, 1550), Rectangle(850, 1800, 1300, 2400)]
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


def test_cone_simu():
    from digicampipe.image.cones_image import ConesImage, cones_simu
    from digicampipe.image.cones_image import simu_match

    # create an image  with a geometry compatible to the camera with
    # known angle, spacing between pixels etc...
    test_image, true_positions = cones_simu(
        offset=(-4.3, -2.1),
        angle_deg=10,
        pixel_radius=35,
    )
    cones_img = ConesImage(test_image)
    cones_img.get_cone(radius_mask=2.1, save_to_file=False)
    cones_img.fit_camera_geometry()
    cones_img.refine_camera_geometry()

    assert simu_match(cones_img, true_positions, std_error_max_px=0.5)


if __name__ == '__main__':
    test_find_stars(method='remote')
    test_cone_simu()
