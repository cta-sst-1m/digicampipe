from digicampipe.image.sky_image import *
import os

def get_pointing():
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
    images_directory = './tests/resources/stars_on_lid'
    image_files = []
    for file in os.listdir(images_directory):
        if file.endswith(".fits"):
            image_files.append(os.path.join(images_directory, file))
    lidccd_obs = LidCCDObservation(image_files, crop_pixels1, crop_pixels2,
                                   scale_low_images_deg=8., scale_high_images_deg=12.)
    lidccd_obs.plot_image_solved('./tests/resources/stars_on_lid')
    lidccd_obs.plot_image_treated('./tests/resources/stars_on_lid')
    lidccd_obs.print_summary()

if __name__ == '__main__':
    get_pointing()