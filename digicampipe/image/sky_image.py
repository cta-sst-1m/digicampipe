from digicampipe.image.kernels import *
from digicampipe.image.utils import *
from astroquery.vizier import Vizier
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from matplotlib.patches import Circle, Rectangle, Arrow
import tempfile
from subprocess import run
import os


class SkyImage(object):
    """SkyImage extract stars from an image and uses astrometry.net to try to get the galactic coordinates.
    To find stars, a spatial high pass filter is used.
    Computation of coordinates is done at construction only if a image_static is given.
    The static_image is removed from the filtered image (to keep only moving stars) prior of using astrometry."""
    def __init__(self, image, image_static=None, threshold=None,
                 scale_low_deg=None, scale_high_deg=None, calculate=False):
        if type(image) is str:
            image = fits.open(self.filename)[0].data
        if type(image) is not np.ndarray:
            raise AttributeError('image must be a filename or a numpy.ndarray')
        # high pass filter
        self.image_stars = signal.convolve2d(image, high_pass_filter_2525, mode='same', boundary='symm')
        self.image_stars[self.image_stars<0] = 0
        # low_pass_filter
        self.image_stars = signal.convolve2d(self.image_stars, gauss(1, (5, 5)), mode='same', boundary='symm')
        self.image_stars[self.image_stars<0] = 0
        self.scale_low_deg = scale_low_deg
        self.scale_high_deg = scale_high_deg
        if image_static is not None:
            self.subtract_static_image(image_static, threshold=threshold)
        self.sources_pixel = None
        self.reference_pixel = None
        self.reference_ra_dec = None
        self.cd_matrix = None
        self.stars_pixel = None
        self.wcs = None
        if calculate:
            self.calculate_galactic_coordinates(scale_low_deg=scale_low_deg, scale_high_deg=scale_high_deg)

    def subtract_static_image(self, image_static, threshold=None ):
        """
        function which subtracts image_static to the filtered image to keep only the moving spikes.
        :param image_static: image subtracted to the filtered image
        :param threshold: pixels with a smaller value than threshold after subtraction are set to 0. Default at 100x the
            average pixel value after subtraction.
        :return:
        """
        if type(image_static) is not np.ndarray:
            raise AttributeError("image_static must be a numpy's ndarray.")
        self.image_stars = self.image_stars - image_static
        self.image_stars[self.image_stars < 0] = 0
        if threshold is None:
            threshold = np.std(self.image_stars) #np.mean(self.image_stars) + 1.5 * np.std(self.image_stars)
        self.image_stars[self.image_stars < threshold] = 0

    def calculate_galactic_coordinates(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            hdu = fits.PrimaryHDU(self.image_stars)
            outfile = os.path.join(tmpdir, 'stars.fits')
            hdu.writeto(outfile, overwrite=True)
            arguments = ['solve-field', outfile, #'--no-background-subtraction', '--no-verify-uniformize',
                         '--ra', str(83.2), '--dec', str(26.2), '--radius', str(10),# '--no-plots',
                         '-t ','--depth', '60']
            if self.scale_low_deg is not None or self.scale_high_deg is not None:
                arguments.append('--scale-units')
                arguments.append('degwidth')
            if self.scale_low_deg is not None:
                arguments.append('--scale-low')
                arguments.append(str(self.scale_low_deg))
            if self.scale_high_deg is not None:
                arguments.append('--scale-high')
                arguments.append(str(self.scale_high_deg))
            run(arguments)
            # sources position in image
            try:
                sources_data = fits.open(os.path.join(tmpdir, 'stars.axy'))[1].data
                self.sources_pixel = np.array((sources_data['X'], sources_data['Y']))
            except FileNotFoundError:
                self.sources_pixel = None
            # coordinate system
            try:
                header_wcs = fits.open(os.path.join(tmpdir, 'stars.wcs'))[0].header
                self.wcs = WCS(header_wcs)
                # reference point
                self.reference_pixel = (header_wcs['CRPIX1'] , header_wcs['CRPIX2'])
                self.reference_ra_dec = (header_wcs['CRVAL1'], header_wcs['CRVAL2'])
                # transformation matrix
                self.cd_matrix = np.array(((header_wcs['CD1_1'], header_wcs['CD1_2']),
                                                             (header_wcs['CD2_1'], header_wcs['CD2_2'])))
            except FileNotFoundError:
                self.reference_pixel = None
                self.reference_ra_dec = None
                self.cd_matrix = None
            # stars position in image
            try:
                stars_data = fits.open(os.path.join(tmpdir, 'stars-indx.xyls'))[1].data
                self.stars_pixel = np.array((stars_data['X'], stars_data['Y']))
            except FileNotFoundError:
                self.stars_pixel = None


class LidCCDImage(object):
    def __init__(self, filename, crop_pixels1, crop_pixels2, image_static=None,
                 threshold=None, scale_low_images_deg=None, scale_high_images_deg=None):
        if type(filename) is not str:
            raise AttributeError('filename must be a string')
        self.filename = filename
        if type(crop_pixels1) is not list:
            raise AttributeError('crop_pixels1 must be a list of pixels')
        if type(crop_pixels2) is not list:
            raise AttributeError('crop_pixels2 must be a list of pixels')
        image=fits.open(self.filename)[0].data
        self.image_shape = image.shape
        self.crop_pixels1 = []
        self.crop_pixels2 = []
        self.sky_images = []
        self.sky_images_shape = []
        print('divide', filename, 'in ', len(crop_pixels1), 'sub-areas')
        for crop_pixel1, crop_pixel2 in zip(crop_pixels1, crop_pixels2):
            image_cropped, crop_pixel1, crop_pixel2 = crop_image(filename, crop_pixel1, crop_pixel2)
            ratios=[self.image_shape[0] / image_cropped.shape[0], self.image_shape[1] / image_cropped.shape[1]]
            scale_low_crop_deg = scale_low_images_deg / np.max(ratios)
            scale_high_crop_deg = scale_high_images_deg / np.min(ratios)
            self.crop_pixels1.append(crop_pixel1)
            self.crop_pixels2.append(crop_pixel2)
            self.sky_images_shape.append(image_cropped.shape)
            sky_image = SkyImage(image_cropped, image_static=image_static, threshold=threshold,
                                 scale_low_deg=scale_low_crop_deg,
                                 scale_high_deg=scale_high_crop_deg)
            self.sky_images.append(sky_image)
        self.center_px = None
        self.center_ra_dec = None
        self.CD = None
        self.combine_coordinates()

    def subtract_static_images(self, static_images, threshold=None):
        for sky_image, static_image in zip(self.sky_images, static_images):
            sky_image.subtract_static_image(static_image, threshold=threshold)

    def calculate_galactic_coordinates(self):
        for sky_image in self.sky_images:
            sky_image.calculate_galactic_coordinates()
        self.combine_coordinates()

    def combine_coordinates(self):
        wcs_list = []
        for sky_image, crop_pixel1 in zip(self.sky_images, self.crop_pixels1):
            if sky_image.reference_ra_dec is not None:
                wcs_list.append(sky_image.wcs)
        if len(wcs_list)>0:
            self.wcs = wcs_list[0]

    def print_summary(self):
        print('matching result for', self.filename)
        for sky_image, crop_pixel1 in zip(self.sky_images, self.crop_pixels1):
            if sky_image.reference_ra_dec is not None:
                print('x pix =',sky_image.reference_pixel[0]+crop_pixel1[0])
                print('y pix =',sky_image.reference_pixel[1]+crop_pixel1[1])
                print('x ra =',sky_image.reference_ra_dec[0])
                print('y de =',sky_image.reference_ra_dec[1])
                print('CD =',sky_image.cd_matrix)

    def plot_image_solved(self, output_dir=None):
        if output_dir is not None:
            plt.ioff()
        fig = plt.figure()
        ax = plt.gca()
        lid_image = fits.open(self.filename)[0].data
        if type(lid_image) is not np.ndarray:
            raise AttributeError([self.filename, ' must be a fit file'])
        plt.imshow(lid_image, cmap='gray')
        for sky_image, crop_pixel1, crop_pixel2 in zip(self.sky_images, self.crop_pixels1, self.crop_pixels2):
            if sky_image.sources_pixel is not None:
                for i in range(min(sky_image.sources_pixel.shape[1], 30)):
                    pixel_x = sky_image.sources_pixel[0, i] + crop_pixel1[0]
                    pixel_y = sky_image.sources_pixel[1, i] + crop_pixel1[1]
                    circle = Circle((pixel_x, pixel_y), radius=9, fill=False, color='r')
                    ax.add_artist(circle)
            if sky_image.stars_pixel is not None:
                for i in range(sky_image.stars_pixel.shape[1]):
                    pixel_x = sky_image.stars_pixel[0, i] + crop_pixel1[0]
                    pixel_y = sky_image.stars_pixel[1, i] + crop_pixel1[1]
                    circle = Circle((pixel_x, pixel_y), radius=11, fill=False, color='g')
                    ax.add_artist(circle)
            width = crop_pixel2[0] - crop_pixel1[0]
            height = crop_pixel2[1] - crop_pixel1[1]
            rect = Rectangle(crop_pixel1, width=width, height=height, fill=False, color='k', linestyle='dashdot')
            ax.add_artist(rect)
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if output_dir is None:
            plt.show()
        else:
            output_filename = self.filename.replace('.fits', '-solved.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')

    def plot_image_treated(self, output_dir=None):
        if output_dir is not None:
            plt.ioff()
        fig = plt.figure()
        ax = plt.gca()
        lid_image = fits.open(self.filename)[0].data
        if type(lid_image) is not np.ndarray:
            raise AttributeError([self.filename, ' must be a fit file'])
        plt.imshow(lid_image, cmap='gray')
        image_treated=np.zeros((self.image_shape))
        vizier_blue = Vizier(columns=['all'], column_filters={"Bjmag": "<12"}, row_limit=-1, catalog=['I/271/out'])
        vizier_red = Vizier(columns=['all'], column_filters={"Rmag": "<12"}, row_limit=-1, catalog=['I/271/out'])
        vizier_green = Vizier(columns=['all'], column_filters={"Vmag": "<12"}, row_limit=-1, catalog=['I/271/out'])
        vizier_gamma = Vizier(columns=['all'], row_limit=-1, catalog=['J/ApJS/197/34'])
        for (sky_image, crop_pixel1, crop_pixel2) in zip(self.sky_images, self.crop_pixels1, self.crop_pixels2):
            image_treated[crop_pixel1[1]:crop_pixel2[1], crop_pixel1[0]:crop_pixel2[0]] = np.log(1+sky_image.image_stars)
            if sky_image.sources_pixel is not None:
                for i in range(min(sky_image.sources_pixel.shape[1], 30)):
                    pixel_x = sky_image.sources_pixel[0, i] + crop_pixel1[0]
                    pixel_y = sky_image.sources_pixel[1, i] + crop_pixel1[1]
                    circle = Circle((pixel_x, pixel_y), radius=20, fill=False, color='w')
                    ax.add_artist(circle)
            if sky_image.wcs is not None:
                self.center_px = np.array((self.image_shape[1], self.image_shape[0])).reshape(1, 2) / 2
                self.center_ra_dec = sky_image.wcs.wcs_pix2world(self.center_px - crop_pixel1, 1)[0]
                print('image center (ra, dec):', self.center_ra_dec)
                center_coordinate = SkyCoord(ra=self.center_ra_dec[0] * u.degree,
                                             dec=self.center_ra_dec[1] * u.degree, frame='icrs')
                result_blue = vizier_blue.query_region(center_coordinate, radius=Angle(5, "deg"),
                                                       catalog=['I/271/out'])[0]
                blue_stars_ra_dec = np.array((result_blue['RA_ICRS_'], result_blue['DE_ICRS_'])).transpose()  # .reshape((-1,2))
                blue_stars_mag = result_blue['Bjmag']
                blue_stars_px = sky_image.wcs.wcs_world2pix(blue_stars_ra_dec, 1)
                for star_px, star_mag in zip(blue_stars_px, blue_stars_mag):
                    if -5 < star_mag <= 12:
                        radius= 18 - star_mag
                    else:
                        radius = 5
                    circle = Circle((star_px[0] + crop_pixel1[0], star_px[1] + crop_pixel1[1]), radius=radius,
                                    fill=False, color='b')
                    ax.add_artist(circle)
                result_red = vizier_red.query_region(center_coordinate, radius=Angle(5, "deg"),
                                                     catalog=['I/271/out'])[0]
                red_stars_ra_dec = np.array((result_red['RA_ICRS_'], result_red['DE_ICRS_'])).transpose()  # .reshape((-1,2))
                red_stars_mag = result_red['Rmag']
                red_stars_px = sky_image.wcs.wcs_world2pix(red_stars_ra_dec, 1)
                for star_px, star_mag in zip(red_stars_px, red_stars_mag):
                    if -5 < star_mag <=12:
                        radius= 18 - star_mag
                    else:
                        radius = 5
                    circle = Circle((star_px[0] + crop_pixel1[0], star_px[1] + crop_pixel1[1]), radius=radius,
                                    fill=False, color='r')
                    ax.add_artist(circle)
                result_green = vizier_green.query_region(center_coordinate, radius=Angle(5, "deg"),
                                                         catalog=['I/271/out'])[0]
                green_stars_ra_dec = np.array(
                    (result_green['RA_ICRS_'], result_green['DE_ICRS_'])).transpose()  # .reshape((-1,2))
                green_stars_mag = result_blue['Vmag']
                green_stars_px = sky_image.wcs.wcs_world2pix(green_stars_ra_dec, 1)
                for star_px, star_mag in zip(green_stars_px, green_stars_mag):
                    if -5 < star_mag <= 12:
                        radius = 18 - star_mag
                    else:
                        radius = 5
                    circle = Circle((star_px[0] + crop_pixel1[0], star_px[1] + crop_pixel1[1]), radius=radius,
                                    fill=False, color='g')
                    ax.add_artist(circle)
                result_gamma = vizier_gamma.query_region(center_coordinate, radius=Angle(5, "deg"),
                                                         catalog=['J/ApJS/197/34'])[0]
                gamma_stars_ra_dec = np.array((result_gamma['RAJ2000'], result_gamma['DEJ2000'])).transpose()  # .reshape((-1,2))
                gamma_stars_name = result_gamma['Name']
                gamma_stars_px = sky_image.wcs.wcs_world2pix(gamma_stars_ra_dec, 1)
                for star_px, gamma_star_name in zip(gamma_stars_px, gamma_stars_name):
                    circle = Circle((star_px[0] + crop_pixel1[0], star_px[1] + crop_pixel1[1]), radius=20,
                                    fill=False, color='Y')
                    ax.add_artist(circle)
                    ax.text(star_px[0] + crop_pixel1[0], star_px[1] + crop_pixel1[1], gamma_star_name, color='Y')
                crab_nebula_px = sky_image.wcs.wcs_world2pix(83.640187, 22.044295, 1)
                circle = Circle((crab_nebula_px[0] + crop_pixel1[0], crab_nebula_px[1] + crop_pixel1[1]), radius=20,
                                fill=False, color='Y')
                ax.add_artist(circle)
                ax.text(crab_nebula_px[0] + crop_pixel1[0], crab_nebula_px[1] + crop_pixel1[1], 'Crab Pulsar', color='Y')
            width = crop_pixel2[0] - crop_pixel1[0]
            height = crop_pixel2[1] - crop_pixel1[1]
            rect = Rectangle(crop_pixel1, width=width, height=height, fill=False, color='y', linestyle='dashdot')
            ax.add_artist(rect)
 #       plt.imshow(image_treated, cmap='gray')
        plt.plot(0,0,'w+')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if output_dir is None:
            plt.show()
        else:
            output_filename = self.filename.replace('.fits', '-treated.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')


class LidCCDObservation:
    def __init__(self, filenames, crop_pixels1, crop_pixels2, threshold=None,
                 scale_low_images_deg=None, scale_high_images_deg=None):
        self.lidccd_images = []
        image_shape = None
        for filename in filenames:
            lidccd_image = LidCCDImage(filename, crop_pixels1, crop_pixels2,
                                       scale_low_images_deg=scale_low_images_deg,
                                       scale_high_images_deg=scale_high_images_deg)
            if image_shape is None:
                image_shape = lidccd_image.image_shape
            elif image_shape != lidccd_image.image_shape:
                raise AttributeError('all images must have the same size')
            self.lidccd_images.append(lidccd_image)
        static_images = []
        for crop_area_index, crop_area_shape in enumerate(lidccd_image.sky_images_shape):
            # get static image for that area
            average_cropped = np.zeros(crop_area_shape)
            for lidccd_image in self.lidccd_images:
                average_cropped += lidccd_image.sky_images[crop_area_index].image_stars / len(self.lidccd_images)
            static_images.append(average_cropped * len(self.lidccd_images) / 4)
        # perform computation of coordinates
        for lidccd_image in self.lidccd_images:
            lidccd_image.subtract_static_images(static_images, threshold=threshold)
            lidccd_image.calculate_galactic_coordinates()

    def print_summary(self):
        for lidccd_image in self.lidccd_images:
            lidccd_image.print_summary()

    def plot_image_solved(self, outpout_dir=None):
        for lidccd_image in self.lidccd_images:
            lidccd_image.plot_image_solved(outpout_dir)

    def plot_image_treated(self, outpout_dir=None):
        for lidccd_image in self.lidccd_images:
            lidccd_image.plot_image_treated(outpout_dir)
