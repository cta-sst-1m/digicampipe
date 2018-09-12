from glob import glob

from digicampipe.image.sky_image import LidCCDObservation
from pkg_resources import resource_filename

from digicampipe.image.lidccd.utils import Rectangle

example_lid_CCD_image_file_paths = glob(
    resource_filename('digicampipe', 'tests/resources/stars_on_lid/*.fits')
)


def get_pointing(method='local'):
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
    lidccd_obs.plot_image_solved()
    lidccd_obs.plot_image_treated()
    lidccd_obs.print_summary()


if __name__ == '__main__':
    get_pointing(method='local')
