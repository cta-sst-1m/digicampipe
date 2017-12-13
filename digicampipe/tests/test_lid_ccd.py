from digicampipe.image.sky_image import *
import os


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
    images_directory = './resources/stars_on_lid'
    image_files = []
    for file in os.listdir(images_directory):
        if file.endswith(".fits"):
            image_files.append(os.path.join(images_directory, file))
    lidccd_obs = LidCCDObservation(image_files, crop_pixels1, crop_pixels2,
                                   scale_low_images_deg=8., scale_high_images_deg=12.)
    nsolved = 0
    for lidccd_image in lidccd_obs.lidccd_images:
        if lidccd_image.wcs is not None:
            nsolved += 1
    assert nsolved > 0


if __name__ == '__main__':
    test_find_stars()
