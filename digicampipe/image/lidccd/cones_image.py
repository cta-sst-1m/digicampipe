import decimal
from decimal import Decimal, ROUND_HALF_EVEN

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from cts_core.camera import Camera
from digicampipe.image.lidccd.kernels import (
    gauss,
    high_pass_filter_77,
    high_pass_filter_2525,
)
from pkg_resources import resource_filename
from scipy import signal, optimize

from digicampipe.image.lidccd.utils import (
    get_neg_hexagonalicity_with_mask,
    set_hexagon,
    get_peaks_separation,
    make_repetitive_mask,
    set_circle,
    reciprocal_to_lattice_space,
    get_consecutive_hex_radius,
    Rectangle,
    CroppedImage,
    fit_gauss_2d,
)
from digicampipe.instrument import geometry

camera_config_file = resource_filename(
    'digicampipe',
    'tests/resources/camera_config.cfg')


class ConesImage(object):
    def __init__(
        self,
        image,
        image_cone=None,
        digicam_config_file=camera_config_file
    ):
        """
        find cones positions from a picture of the photo detection plane.
        Usage :
            1. create ConesImage,
            2. extract a cone with get_cone(),
            3. get cones position using fit_camera_geometry()
            4. (optional) refine positions using refine_camera_geometry()

        Parameters
        ----------
        image : 2d-array
            array containing the lid CCD image.
        image_cone :
            optional fit filename of the cone image.
            the fit file is created calling get_cone()
        digicam_config_file :
            path to the digicam configuration file
        """
        self.pixels_nvs = get_pixel_nvs(digicam_config_file)
        # high pass filter
        self.image_cones = signal.convolve2d(
            image, high_pass_filter_77, mode='same', boundary='symm')
        # low_pass_filter
        self.image_cones = signal.convolve2d(
            self.image_cones,
            gauss(3, (15, 15)),
            mode='same',
            boundary='symm')
        # thresholds
        self.image_cones[self.image_cones < 0] = 0
        self.image_cones[self.image_cones > np.std(self.image_cones)] = 0
        self.image_cones -= np.mean(self.image_cones)
        # fft
        self.fft_image_cones = np.fft.fft2(self.image_cones)
        self.center_fitted = None
        # filled by get_cones_separation_reciprocal():
        self.ks = None  # distance between fft peaks
        self.v1_lattice = None  # distance between 2 hexagons

        # distance between 2 hexagons (after v1 in anticlockwise)
        self.v2_lattice = None
        # distance between 2 hexagons (after v2 in anticlockwise)
        self.v3_lattice = None
        self.r1 = None  # radius of the hexagon pixel
        self.r2 = None  # radius of the hexagon (after r1 in anticlockwise)
        self.r3 = None  # radius of the hexagon (after r2 in anticlockwise)
        # position of all pixel predicted from fit,
        # call plot_cones_presence() to compute it.
        self.pixels_pos_predict = None
        # cone position fitted, call fit_camera_geometry() to compute it
        self.pixels_fit_px = None
        self.cone_presence = None  # convolution of cone image in lid ccd image
        # individual cone image, call get_cone() function to compute it
        if image_cone is None:
            self.image_cone = None
        else:
            hdu = fits.open(image_cone)[0]
            self.image_cone = hdu.data
            self.center_fitted = np.array((
                np.real(hdu.header['center']),
                np.imag(hdu.header['center'])
            ))
            center = (np.array(self.image_cones.shape[::-1]) - 1) / 2
            print(
                'center of loaded cone at',
                self.center_fitted,
                ',',
                self.center_fitted - center,
                'from center')
            ks1 = np.array((
                np.real(hdu.header['ks1']),
                np.imag(hdu.header['ks1'])))
            ks2 = np.array((
                np.real(hdu.header['ks2']),
                np.imag(hdu.header['ks2'])))
            self.ks = np.array((ks1, ks2))
            self.v1_lattice = np.array((
                np.real(hdu.header['v1']), np.imag(hdu.header['v1'])))
            self.v2_lattice = np.array((
                np.real(hdu.header['v2']), np.imag(hdu.header['v2'])))
            self.v3_lattice = np.array((
                np.real(hdu.header['v3']), np.imag(hdu.header['v3'])))
            self.r1 = np.array((
                np.real(hdu.header['r1']), np.imag(hdu.header['r1'])))
            self.r2 = np.array((
                np.real(hdu.header['r2']), np.imag(hdu.header['r2'])))
            self.r3 = np.array((
                np.real(hdu.header['r3']), np.imag(hdu.header['r3'])))

    def plot_cones(self, output_filename=None, radius_mask=None):
        """ plot the lid CCD image after filtering

        If radius_mask is None
        otherwise plot the cones contents of the image.

        The cones contents is obtained applying a mask of radius radius_mask
        around each peak of the FFT and doing the
        inverse FFT transformation.

        Parameter
        ---------
        radius_mask :
            optional radius of the mask used for each peak in the FFT
        output_filename :
            path where to put the resulting image.
        """
        if output_filename is None:
            if radius_mask is None:
                output_filename='cones.png'
            else:
                output_filename='cones-filtered.png'
        plt.ioff()
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = plt.gca()
        if radius_mask is not None:
            mask = self.get_fft_mask(radius=radius_mask)
            image_cones = np.real(np.fft.ifft2(self.fft_image_cones * mask))
            image_cones -= np.min(image_cones)
        else:
            image_cones = self.image_cones
        plt.imshow(image_cones, cmap='gray')
        plt.autoscale(False)
        if radius_mask is not None:
            if self.center_fitted is None:
                center = (np.array(self.image_cones.shape)[::-1] - 1) / 2
            else:
                center = self.center_fitted
            v1_points = np.array((np.zeros((2,)), self.v1_lattice)) + center
            v2_points = np.array((np.zeros((2,)), self.v2_lattice)) + center
            r1_points = np.array((np.zeros((2,)), self.r1)) + center
            r2_points = np.array((np.zeros((2,)), self.r2)) + center
            r3_points = np.array((np.zeros((2,)), self.r3)) + center
            plt.plot(v1_points[:, 0], v1_points[:, 1], 'b--', linewidth=1)
            plt.plot(v2_points[:, 0], v2_points[:, 1], 'r--', linewidth=1)
            plt.plot(r1_points[:, 0], r1_points[:, 1], 'b-', linewidth=1)
            plt.plot(r2_points[:, 0], r2_points[:, 1], 'r-', linewidth=1)
            plt.plot(r3_points[:, 0], r3_points[:, 1], 'g-', linewidth=1)
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(output_filename, 'saved.')

    def scan_cone_position(
        self,
        radius_mask,
        output_filename='hexagonalicity.png',
        center_scan=None,
        rotations=(60, 300)
    ):
        """
        Calculate the hexagonalicity for each pixel inside a camera's pixel.

        Parameter
        ---------
        radius_mask:
            radius of the mask used for extracting
            pixels from image (see plot_cones())
        output_filename:
            path where to put the resulting image.
        center_scan:
            optional position of the center of the camera's pixel.
            Center of image is used if None (default)
        rotations:
            angles in degrees used for calculating hexagonalicity.

        remark :  output should look like
        http://n.ethz.ch/~nielssi/download/4.%20Semester/AC%20II/Unterlagen/symmetry_2D_3.pdf page 22
        """
        if type(radius_mask) is not float:
            raise AttributeError('radius_mask must be a float.')
        if self.r1 is None or self.r2 is None:
            raise AttributeError(
                'camera pixel geometry must be computed '
                'prior of calling scan_cone_position().')
        mask = self.get_fft_mask(radius=radius_mask)
        image_cones = np.real(np.fft.ifft2(self.fft_image_cones * mask))
        center_image = (np.array(image_cones.shape[::-1]) - 1) / 2
        if center_scan is None:
            center_scan = center_image
        scan_area = np.zeros(image_cones.shape, dtype=bool)
        scan_area = set_hexagon(
            scan_area, center_scan, r1=self.r1, r2=self.r2, value=1)
        scan_result = np.zeros(image_cones.shape, dtype=float)
        all_pixels_y, all_pixels_x = np.indices(scan_area.shape)
        pixels_x = all_pixels_x[scan_area == 1].flatten()
        pixels_y = all_pixels_y[scan_area == 1].flatten()
        npixel = pixels_x.shape[0]
        print('calculating hexagonalicity for each position in the pixel:')
        last_precent = 0
        for pixel_x, pixel_y, iter in zip(
                pixels_x[::1],
                pixels_y[::1],
                range(npixel)
        ):
            percent_done = np.floor((iter+1)*100/npixel)
            if percent_done > last_precent:
                print(percent_done, '%')
                last_precent = percent_done
            hexagonalicity = -get_neg_hexagonalicity_with_mask(
                (pixel_x, pixel_y),
                image_cones,
                self.r1,
                self.r2,
                rotations=rotations
            )
            scan_result[pixel_y, pixel_x] = hexagonalicity
        plt.ioff()
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = plt.subplot(1, 2, 2)
        vmin = np.min(scan_result[scan_result > 0])
        vmax = np.max(scan_result[scan_result > 0])
        plt.imshow(scan_result, cmap='gray', vmin=vmin, vmax=vmax)
        plt.autoscale(False)
        max_y, max_x = np.unravel_index(
            np.argmax(scan_result),
            dims=scan_result.shape
        )
        plt.plot(max_x, max_y, 'r+')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.xlim([np.min(pixels_x)-0.5, np.max(pixels_x)+0.5])
        plt.ylim([np.min(pixels_y)-0.5, np.max(pixels_y)+0.5])
        ax = plt.subplot(1, 2, 1)
        plt.imshow(image_cones * scan_area, cmap='gray')
        plt.plot(max_x, max_y, 'r+')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(output_filename, 'saved.')

    def get_cone(self, radius_mask, output_filename=None, cone_filename=None):
        """find the center of a camera's pixel.
        The pixel image along with important parameter can be save for future use
        using the cone_filename parameter.
        Parameter
        ---------
        radius_mask :
            radius of the mask used for extracting pixels
            from image (see plot_cones())
        output_filename : optional,
            path where to put the resulting cone image.
        cone_filename : optional, name of the fits file where to save
            the cone image along with important parameters.
        """
        if type(radius_mask) is not float:
            raise AttributeError('radius_mask must be a float.')
        mask_fft = self.get_fft_mask(radius=radius_mask)
        image_cones = np.real(np.fft.ifft2(self.fft_image_cones * mask_fft))
        center_image = (np.array(image_cones.shape[::-1]) - 1) / 2
        if self.center_fitted is None:
            center_tested = center_image
        else:
            center_tested = self.center_fitted
        points = np.array((center_tested + self.r1,
                           center_tested + self.r2,
                           center_tested + self.r3,
                           center_tested - self.r1,
                           center_tested - self.r2,
                           center_tested - self.r3))
        bounds = ((np.min(points[:, 0]), np.max(points[:, 0])),
                  (np.min(points[:, 1]), np.max(points[:, 1])))
        offsets = (
            0*self.r1,
            (self.r1+self.r2)/3,
            self.r1,
            self.r2,
            (self.r2+self.r3)/3,
            self.r3,
            (self.r3-self.r1)/3,
            -self.r1,
            (-self.r1-self.r2)/3,
            -self.r2,
            (-self.r2-self.r3)/3,
            -self.r3,
            (-self.r3+self.r1)/3
        )
        results = []
        for i, offset in enumerate(offsets):
            res = optimize.minimize(
                get_neg_hexagonalicity_with_mask,
                center_tested + offset,
                args=(image_cones, self.r1, self.r2, (60, 300)),
                bounds=bounds,
                method='TNC',
                options={
                    'disp': False,
                    'eps': 1,
                    'xtol': 1e-1,
                    'maxiter': 200}
                )
            results.append(res)
            print(
                'fit', i+1, '/', len(offsets), 'at',
                res.x, 'done: hex=', -res.fun)
            if -res.fun > 0.85:
                break
        print('refine fit')
        hex_results = np.array([-res.fun for res in results])
        pos_results = np.array([res.x for res in results])
        # fine fit for best result
        res = optimize.minimize(
            get_neg_hexagonalicity_with_mask,
            pos_results[np.argmax(hex_results)],
            args=(image_cones, 6*self.r1, 6*self.r2, (60, 120, 180, 240, 300)),
            bounds=bounds,
            method='TNC',
            options={
                'disp': False,
                'eps': .1,
                'xtol': 1e-5,
                'maxiter': 2000}
        )
        if -res.fun > 0.75:
            self.center_fitted = res.x
            hex_result = -res.fun
        else:
            print('WARNING: hexagonalicity is small.')
            results.append(res)
            hex_results = np.array([-res.fun for res in results])
            pos_results = np.array([res.x for res in results])
            self.center_fitted = pos_results[np.argmax(hex_results)]
            hex_result = hex_results[np.argmax(hex_results)]
        print(
            'pixel center found: ',
            self.center_fitted,
            'with hex=',
            hex_result)
        print(self.center_fitted - center_image, 'px from center')
        points_fitted = points - center_tested + np.array(self.center_fitted)
        pixels_x_min = int(np.floor(np.min(points_fitted[:, 0])))
        pixels_x_max = int(np.ceil(np.max(points_fitted[:, 0]))) + 1
        pixels_y_min = int(np.floor(np.min(points_fitted[:, 1])))
        pixels_y_max = int(np.ceil(np.max(points_fitted[:, 1]))) + 1
        # plot_image(image_cones)
        image_crop = image_cones[
            pixels_y_min:pixels_y_max,
            pixels_x_min:pixels_x_max]
        # plot_image(image_crop)
        center_crop = (
            np.array(self.center_fitted) -
            np.array((pixels_x_min, pixels_y_min))
        )
        mask_hexa = np.zeros_like(image_crop)
        mask_hexa = set_hexagon(
            mask_hexa, center=center_crop, r1=self.r1, r2=self.r2)
        self.image_cone = image_crop * mask_hexa
        if cone_filename is not None:
            hdu = fits.PrimaryHDU(self.image_cone)
            hdu.header['center'] = (
                self.center_fitted[0] + 1j * self.center_fitted[1],
                'fitted position (in original image) of the hexagon center')
            hdu.header['ks1'] = (
                self.ks[0, 0] + 1j * self.ks[0, 1],
                (
                    '1st vector of the base of the hexagonal lattice'
                    ' in reciprocal space'))
            hdu.header['ks2'] = (
                self.ks[1, 0] + 1j * self.ks[1, 1],
                (
                    '2nd vector of the base of the hexagonal lattice '
                    'in reciprocal space'))
            hdu.header['v1'] = (
                self.v1_lattice[0] + 1j * self.v1_lattice[1],
                'spacing between 2 hexagons along the 1st axis')
            hdu.header['v2'] = (
                self.v2_lattice[0] + 1j * self.v2_lattice[1],
                'spacing between 2 hexagons along the 2nd axis')
            hdu.header['v3'] = (
                self.v3_lattice[0] + 1j * self.v3_lattice[1],
                'spacing between 2 hexagons along the 3rd axis')
            hdu.header['r1'] = (
                self.r1[0] + 1j * self.r1[1],
                '1st radius of the hexagon')
            hdu.header['r2'] = (
                self.r2[0] + 1j * self.r2[1],
                '2nd radius of the hexagon')
            hdu.header['r3'] = (
                self.r3[0] + 1j * self.r3[1],
                '3rd radius of the hexagon')
            hdu.writeto(cone_filename, overwrite=True)
            print('cone saved to ', cone_filename)
        if output_filename is not None:
            plt.ioff()
            fig = plt.figure(figsize=(8, 6), dpi=600)
            ax = plt.gca()
            plt.imshow(self.image_cone, cmap='gray')
            plt.autoscale(False)
            plt.plot(
                self.center_fitted[0]-pixels_x_min,
                self.center_fitted[1]-pixels_y_min,
                'y+'
            )
            plt.grid(None)
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')

    def get_cones_presence(self):
        """
        To calculate the convolution of cone image on filtered lid CCD image.
        """
        if self.image_cone is None:
            raise decimal.InvalidOperation(
                'cone image not determined, call get_cone()'
                ' before get_cones_presence().')
        self.cone_presence = signal.fftconvolve(
            self.image_cones, self.image_cone, mode='same')
        self.cone_presence = signal.fftconvolve(
            self.cone_presence, high_pass_filter_2525, mode='same')
        self.cone_presence[self.cone_presence < 0] = 0

    def plot_fft_cones(self, output_filename=None, radius_mask=None):
        """plot the FFT of the filtered lid CCD image.

        If radius_mask is given, everything more distant than it from
        a peak is set to 0.
        radius_mask :
            radius in pixels of circular mask around each peak.
            Default = no mask.
        output_filename :
            path where to put the resulting image.
            default to 'cones-fft.png' without mask or
            'cones-fft-masked.png' with mask
        """
        plt.ioff()
        if output_filename is None:
            if radius_mask is None:
                output_filename='cones-fft.png'
            else:
                output_filename='cones-fft-masked.png'
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = fig.gca()
        if radius_mask is not None:
            mask = self.get_fft_mask(radius=radius_mask)
            fft_image_cones = self.fft_image_cones * mask
        else:
            fft_image_cones = self.fft_image_cones
        plt.imshow(
            np.log(1 + np.abs(np.fft.fftshift(fft_image_cones))), cmap='gray')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(output_filename, 'saved.')

    def get_fft_mask(self, radius=3):
        """create a mask to keep only pixels closer to peaks than radius.

        Parameter
        ---------
        radius :
            radius in pixels of circular mask around each peak.

        Returns
        -------
            the mask to be applied on fft
            (image of same size as self.fft_image_cones).
        """
        if self.ks is None:
            self.get_cones_separation_reciprocal()
        center = (np.array((self.fft_image_cones.shape[::-1])) - 1) / 2
        mask = make_repetitive_mask(
            self.fft_image_cones.shape,
            radius,
            self.ks[0, :],
            self.ks[1, :],
            center,
            nrepetition=100
        )
        for nk1 in range(-50, 50):
            for nk2 in range(-50, 50):
                center_circle = (
                    center +
                    nk1 * self.ks[0, :] +
                    nk2 * self.ks[1, :]
                )
                mask = set_circle(
                    mask,
                    center=center_circle.reshape((2,)),
                    radius=radius,
                    value=1)
        return np.fft.ifftshift(mask)

    def get_cones_separation_reciprocal(self, output_filename=None):
        """calculate the distance between 2 neighbour pixels

        (as vectors) and 3 consecutive radius.
        The distances are stored in self.v[123]_lattice
        and the radius are stored in self.r[123]

        output_filename :
            optional path where to put the
            auto-correlation of the FFT image.
        """
        fft_image_cones_shifted = signal.convolve2d(
            np.fft.fftshift(np.abs(self.fft_image_cones)),
            high_pass_filter_77,
            mode='same',
            boundary='wrap')
        fft_image_cones_shifted[fft_image_cones_shifted < 0] = 0
        ks_base, auto_correlation_saved, center_peaks = get_peaks_separation(
            fft_image_cones_shifted,
            crop_range=800)
        if output_filename is not None:
            plt.ioff()
            fig = plt.figure(figsize=(8, 6), dpi=600)
            plt.imshow(auto_correlation_saved, cmap='gray')
            plt.autoscale(False)
            plt.plot(center_peaks[:, 0], center_peaks[:, 1], 'y+')
            k1_points = np.array(
                (np.zeros((2,)), ks_base[0, :])) + center_peaks[0, :]
            k2_points = np.array(
                (np.zeros((2,)), ks_base[1, :])) + center_peaks[0, :]
            plt.plot(k1_points[:, 0], k1_points[:, 1], 'b-', linewidth=1)
            plt.plot(k2_points[:, 0], k2_points[:, 1], 'r-', linewidth=1)
            plt.grid(None)
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')
        self.ks = ks_base
        v1, v2 = reciprocal_to_lattice_space(self.ks[0, :], self.ks[1, :],
                                             self.fft_image_cones.shape[::-1])
        self.v1_lattice, self.v2_lattice = get_consecutive_hex_radius(v1, v2)
        self.v3_lattice = self.v2_lattice - self.v1_lattice
        print(
            "v1=", self.v1_lattice,
            "|v1|=", np.abs(self.v1_lattice[0] + 1j * self.v1_lattice[1]))
        print(
            "v2=", self.v2_lattice,
            "|v2|=", np.abs(self.v2_lattice[0] + 1j * self.v2_lattice[1]))
        print(
            "v3=", self.v3_lattice,
            "|v3|=", np.abs(self.v3_lattice[0] + 1j * self.v3_lattice[1]))
        self.r1 = (self.v1_lattice + self.v2_lattice) / 3
        self.r2 = self.v2_lattice - self.r1
        self.r3 = self.v3_lattice - self.r2
        print("r1=", self.r1, "|r1|=", np.abs(self.r1[0] + 1j * self.r1[1]))
        print("r2=", self.r2, "|r2|=", np.abs(self.r2[0] + 1j * self.r2[1]))
        print("r3=", self.r3, "|r3|=", np.abs(self.r3[0] + 1j * self.r3[1]))

    def plot_cones_presence(self, output_filename='cones-presence.png'):
        """
        Plot convolution of cone image on filtered lid CCD image.
        :param output_filename: directory where to put the resulting images.
            default to 'cones-presence.png'
        """
        if self.cone_presence is None:
            self.get_cones_presence()
        plt.ioff()
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = plt.gca()
        plt.imshow(self.cone_presence, cmap='gray')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(output_filename, 'saved.')

    def fit_pixels_position(self, radius_mask, sigma_peak=5, offset_max=5):
        self.pixels_fit_px = []
        nfail = 0
        for i1 in range(-100, 100):
            for i2 in range(-100, 100):
                peak_pos_aprox = (
                    self.center_fitted +
                    i1 * self.v1_lattice +
                    i2 * self.v2_lattice)
                crop_px1 = np.floor(peak_pos_aprox - radius_mask).astype(int)
                crop_px2 = np.ceil(peak_pos_aprox + radius_mask).astype(int)
                rectangle = Rectangle(
                    left=crop_px1[0],
                    bottom=crop_px1[1],
                    right=crop_px2[0],
                    top=crop_px2[1])
                try:
                    crop = CroppedImage(
                        self.cone_presence,
                        rectangle=rectangle,
                        strict_limit=False)
                except AttributeError:
                    continue  # expected peak position outside of the image
                if (
                    np.any([
                        crop.rectangle.width(), crop.rectangle.height()
                    ] <= np.round(radius_mask))
                ):
                    continue  # cropped area too small
                crop_center = (np.array(crop.image.shape[::-1]) - 1) / 2
                max_pos_crop = np.argmax(crop.image)
                [max_pos_y, max_pos_x] = np.unravel_index(
                    max_pos_crop, crop.image.shape)
                init_amplitude = (
                    crop.image[max_pos_y, max_pos_x] - np.min(crop.image))
                init_param = (
                    init_amplitude,
                    max_pos_x,
                    max_pos_y,
                    sigma_peak,
                    sigma_peak,
                    0,
                    np.min(crop.image)
                )
                fit_result, success = fit_gauss_2d(
                    crop.image.transpose(), initial_param=init_param)
                (
                    amplitude,
                    xcenter,
                    ycenter,
                    xsigma,
                    ysigma,
                    rot,
                    bkg
                ) = fit_result

                if (
                    0 < success <= 4 and
                    0 < xsigma < 2*sigma_peak and
                    0 < ysigma < 2*sigma_peak and
                    np.abs(xcenter - crop_center[0]) < offset_max and
                    np.abs(ycenter - crop_center[1]) < offset_max
                ):
                    self.pixels_fit_px.append(
                        np.array([xcenter, ycenter]) + crop_px1)

                else:
                    nfail += 1
                if np.mod(len(self.pixels_fit_px) + nfail, 100) == 0:
                    print(
                        len(self.pixels_fit_px) + nfail,
                        'fits done, (',
                        len(self.pixels_fit_px), 'successful )'
                    )
        self.pixels_fit_px = np.array(self.pixels_fit_px)

    def fit_camera_geometry(
        self,
        radius_mask=15.1,
        sigma_peak=4,
        offset_max=3
    ):
        """
        Fit peaks positions in the convolution of cone image
        on filtered lid CCD image. Find the best orientation and position for
        the camera geometry to match the peaks.

        Parameter
        ---------

        radius_mask :
            radius around expected peak used in peak fitting
        sigma_peak :
            estimated sigma_peak of peaks. Used as initial value used in fit.
            Also, if the fitted
            sigma is larger than 2x this value, the peak is discarded.
        offset_max :
            maximum offset of the peak with respect to the expectied position.
            If the fitted offset is large, the peak is discarded.
        """
        print("find pixels in lid CCD image")
        if self.cone_presence is None:
            self.get_cones_presence()
        if self.pixels_fit_px is None:
            self.fit_pixels_position(
                radius_mask=radius_mask,
                sigma_peak=sigma_peak,
                offset_max=offset_max)
        print('look for pixels geometry:')
        nv_prec = Decimal('1')
        pixels_nvs_dec = [
            [
                Decimal(n1*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3
            ]
            for n1, n2 in self.pixels_nvs.transpose()
        ]
        pixels_nvs_set = set(map(tuple, pixels_nvs_dec))
        best_matching_nvs = 0
        best_match_nv1, best_match_nv2 = None, None
        best_match_v1, best_match_v2 = None, None
        best_match_r = None
        for v1_test, v2_test in [
            [self.v1_lattice, self.v2_lattice],
            [self.v2_lattice, self.v3_lattice]
        ]:
            v_matrix = np.array([v1_test, v2_test]).transpose()
            for r in self.r1, self.r2:
                for nv1 in range(-10, 10):
                    for nv2 in range(-10, 10):
                        pixels_fit_px_shifted = (
                            self.pixels_fit_px -
                            self.center_fitted -
                            nv1 * v1_test -
                            nv2 * v2_test -
                            r
                        )
                        pixels_fit_nvs = np.linalg.pinv(v_matrix).dot(
                            pixels_fit_px_shifted.transpose())
                        pixels_fit_nvs_dec = [
                            [
                                Decimal(n1*3).quantize(
                                    nv_prec,
                                    rounding=ROUND_HALF_EVEN
                                ) / 3,
                                Decimal(n2*3).quantize(
                                    nv_prec,
                                    rounding=ROUND_HALF_EVEN
                                ) / 3
                            ] for n1, n2 in pixels_fit_nvs.transpose()
                        ]
                        pixels_fit_nvs_set = set(
                            map(tuple, pixels_fit_nvs_dec))
                        matching_nvs = len(
                            pixels_fit_nvs_set.intersection(pixels_nvs_set))
                        if matching_nvs > best_matching_nvs:
                            best_matching_nvs = matching_nvs
                            best_match_nv1, best_match_nv2 = nv1, nv2
                            best_match_v1, best_match_v2 = v1_test, v2_test
                            best_match_r = r
                        if matching_nvs == 1296:
                            print('perfect solution: nv1=', nv1, 'nv2=', nv2,
                                  'v1=', v1_test, 'v2=', v2_test, 'r=', r)
        self.center_fitted += (
            best_match_nv1 * best_match_v1 +
            best_match_nv2 * best_match_v2 +
            best_match_r)
        self.v1_lattice = best_match_v1
        self.v2_lattice = best_match_v2
        self.v3_lattice = self.v2_lattice - self.v1_lattice
        self.r1 = (self.v1_lattice + self.v2_lattice) / 3
        self.r2 = self.v2_lattice - self.r1
        self.r3 = self.v3_lattice - self.r2
        v_matrix = np.array([
            self.v1_lattice,
            self.v2_lattice,
            self.center_fitted]
        ).transpose()
        self.pixels_pos_predict = v_matrix.dot(
            np.vstack(
                (
                    self.pixels_nvs,
                    np.ones(
                        (
                            1,
                            self.pixels_nvs.shape[1]
                        )
                    )
                )
            )
        )
        print(
            'best base: v1=',
            self.v1_lattice,
            ', v2=',
            self.v2_lattice,
            'v3=',
            self.v2_lattice,
            'r=',
            best_match_r)
        print('best match: nv1=', best_match_nv1, ', nv2=', best_match_nv2)
        pixels_fit_nvs = np.linalg.pinv(
            v_matrix[:, 0:2]
        ).dot(
            (self.pixels_fit_px - self.center_fitted).transpose()
        )
        pixels_fit_nvs_dec = [
            [
                Decimal(n1*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3
            ]
            for n1, n2 in pixels_fit_nvs.transpose()]
        pixels_fit_nvs_set = set(map(tuple, pixels_fit_nvs_dec))
        nvs_fit_matching_set = pixels_fit_nvs_set.intersection(pixels_nvs_set)
        if len(nvs_fit_matching_set) != best_matching_nvs:
            print(
                'ERROR reproducing best matching solution: got',
                len(nvs_fit_matching_set),
                'matches instead of',
                best_matching_nvs)
            return
        is_fit_matching = np.array(
            [tuple(nv) in nvs_fit_matching_set for nv in pixels_fit_nvs_dec])
        fit_not_matching = is_fit_matching == 0
        print(
            np.sum(is_fit_matching), 'fits in best match,',
            np.sum(fit_not_matching), 'outside')

    def refine_camera_geometry(self):
        """
        Refit the distance between pixels only taking into account the matching peaks.
        """
        nv_prec = Decimal('0.1')
        v_matrix = np.array([self.v1_lattice, self.v2_lattice]).transpose()
        pixels_fit_nvs = np.linalg.pinv(v_matrix).dot((self.pixels_fit_px - self.center_fitted).transpose())
        pixels_fit_nvs_dec = [
            [
                Decimal(n1*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3
            ]
            for n1, n2 in pixels_fit_nvs.transpose()
        ]
        pixels_fit_nvs_set = set(map(tuple, pixels_fit_nvs_dec))
        pixels_nvs_dec = [
            [
                Decimal(n1*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3
            ]
            for n1, n2 in self.pixels_nvs.transpose()
        ]
        pixels_nvs_set = set(map(tuple, pixels_nvs_dec))
        nvs_fit_matching_set = pixels_fit_nvs_set.intersection(pixels_nvs_set)
        is_fit_matching = np.array(
            [tuple(nv) in nvs_fit_matching_set for nv in pixels_fit_nvs_dec])
        print('global fit of lattice base vectors...')
        n_matching = np.sum(is_fit_matching)
        nvs = np.vstack(
            (pixels_fit_nvs[:, is_fit_matching], np.ones((1, n_matching))))
        pxs = self.pixels_fit_px[is_fit_matching, :].transpose()
        precise_vs = pxs.dot(np.linalg.pinv(nvs))
        self.v1_lattice = precise_vs[:, 0]
        self.v2_lattice = precise_vs[:, 1]
        self.v3_lattice = self.v2_lattice - self.v1_lattice
        self.r1 = (self.v1_lattice + self.v2_lattice) / 3
        self.r2 = self.v2_lattice - self.r1
        self.r3 = self.v3_lattice - self.r2
        self.center_fitted = precise_vs[:, 2]
        self.pixels_pos_predict = precise_vs.dot(np.vstack((self.pixels_nvs, np.ones((1, self.pixels_nvs.shape[1])))))
        print(
            "v1=", self.v1_lattice,
            "|v1|=", np.abs(self.v1_lattice[0] + 1j * self.v1_lattice[1]))
        print(
            "v2=", self.v2_lattice,
            "|v2|=", np.abs(self.v2_lattice[0] + 1j * self.v2_lattice[1]))
        print(
            "v3=", self.v3_lattice,
            "|v3|=", np.abs(self.v3_lattice[0] + 1j * self.v3_lattice[1]))
        print(
            "r1=", self.r1,
            "|r1|=", np.abs(self.r1[0] + 1j * self.r1[1]))
        print(
            "r2=", self.r2,
            "|r2|=", np.abs(self.r2[0] + 1j * self.r2[1]))
        print(
            "r3=", self.r3,
            "|r3|=", np.abs(self.r3[0] + 1j * self.r3[1]))
        center_image = (np.array(self.image_cones.shape[::-1]) - 1) / 2
        print(
            "center=", self.center_fitted,
            ',', self.center_fitted - center_image, 'from center')

    def plot_camera_geometry(self, output_filename='cones-presence-filtered.png'):
        """
        Plot the lid CCD image along with the camera geometry:
        - blue cross are shown for each of the detected pixel which match
        the best camera geometry found.
        - yellow cross are shown for each of the detected pixel which does not match
        the best camera geometry found.
        - green cross are shown for each pixel of the best camera geometry found
        with a detected pixel
        - red cross are shown for each pixel of the best camera geometry found
        without a detected pixel
        :param output_filename: path for the resulting image
        """
        nv_prec = Decimal('1')
        v_matrix = np.array([self.v1_lattice, self.v2_lattice]).transpose()
        pixels_fit_nvs = np.linalg.pinv(v_matrix).dot(
            (self.pixels_fit_px - self.center_fitted).transpose()
        )
        pixels_fit_nvs_dec = [
            [
                Decimal(n1*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3
            ]
            for n1, n2 in pixels_fit_nvs.transpose()
        ]
        pixels_fit_nvs_set = set(map(tuple, pixels_fit_nvs_dec))
        pixels_nvs_dec = [
            [
                Decimal(n1*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(nv_prec, rounding=ROUND_HALF_EVEN)/3
            ]
            for n1, n2 in self.pixels_nvs.transpose()
        ]
        pixels_nvs_set = set(map(tuple, pixels_nvs_dec))
        nvs_fit_matching_set = pixels_fit_nvs_set.intersection(pixels_nvs_set)
        is_fit_matching = np.array(
            [tuple(nv) in nvs_fit_matching_set for nv in pixels_fit_nvs_dec])
        fit_not_matching = is_fit_matching == 0
        is_pixel_fitted = np.array(
            [tuple(nv) in nvs_fit_matching_set for nv in pixels_nvs_dec])
        pixels_not_fitted = is_pixel_fitted == 0
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = plt.gca()
        plt.imshow(self.image_cones, cmap='gray')
        # plt.imshow(self.cone_presence, cmap='gray')
        plt.autoscale(False)
        plt.plot(
            self.pixels_pos_predict[0, is_pixel_fitted],
            self.pixels_pos_predict[1, is_pixel_fitted], 'gx', ms=5, mew=0.2)
        plt.plot(
            self.pixels_pos_predict[0, pixels_not_fitted],
            self.pixels_pos_predict[1, pixels_not_fitted], 'rx', ms=5, mew=0.2)
        plt.plot(
            self.pixels_fit_px[is_fit_matching, 0],
            self.pixels_fit_px[is_fit_matching, 1], 'b+', ms=5, mew=0.2)
        plt.plot(
            self.pixels_fit_px[fit_not_matching, 0],
            self.pixels_fit_px[fit_not_matching, 1], 'y+', ms=5, mew=0.2)
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(output_filename, 'saved.')


def simu_match(cones_image, true_positions, std_error_max_px=0.5):
    if cones_image.pixels_pos_predict is None:
        cones_image.fit_camera_geometry()
    # as camera is invariant by 60 deg rotation, we try the 3 possibilities:
    diffs = []
    offsets = []
    for i in range(3):
        angle = i * 2 / 3 * np.pi
        R = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]])
        pos_predict = R.dot(
            cones_image.pixels_pos_predict -
            cones_image.center_fitted.reshape(2, 1)
        ) + cones_image.center_fitted.reshape(2, 1)
        diffs.append(np.std(pos_predict - true_positions))
        offsets.append(np.mean(pos_predict - true_positions, axis=1))
    print('error on pixel position: ', np.min(diffs))
    print('offset=', offsets[np.argmin(diffs)])
    angle = np.argmin(diffs) * 2 / 3 * np.pi
    R = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    cones_image.pixels_pos_predict = R.dot(
        cones_image.pixels_pos_predict -
        cones_image.center_fitted.reshape(2, 1)
    ) + cones_image.center_fitted.reshape(2, 1)
    error = np.std(cones_image.pixels_pos_predict - true_positions)
    return error < std_error_max_px


def cones_simu(
    pixels_nvs=None,
    offset=(0, 0),
    angle_deg=0,
    image_shape=(2472, 3296),
    pixel_radius=38.3,
    noise_ampl=0.,
    output_filename=None,
):
    """
    function to create a test cones image according to given parameters

    Parameters
    ----------

    offset:
        offest of the center of the camera geometry with respect
        to the center of the test image
    angle_deg :
        angle of the camera geometry
    image_shape :
        shape of the test image (npixel_y, npixel_x)
    pixel_radius:
        length of the radius of an hexagonal pixel
        (from the center of the hexagon to the border)
    noise_ampl :
        amplitude of random noise added to the test image (gaussian)
    output_filename:
        optional path where to put the original lid CCD image
    :return:
    """
    if pixels_nvs is None:
        pixels_nvs = get_pixel_nvs()
    angle_rot = angle_deg / 180 * np.pi
    offset = np.array(offset)
    image = np.zeros(image_shape)
    center = (np.array(image.shape[::-1]) - 1) / 2
    r1 = pixel_radius * np.array((np.cos(angle_rot), np.sin(angle_rot)))
    r2 = pixel_radius * np.array(
        (np.cos(np.pi / 3 + angle_rot), np.sin(np.pi / 3 + angle_rot)))
    r3 = pixel_radius * np.array(
        (np.cos(np.pi * 2 / 3 + angle_rot), np.sin(np.pi * 2 / 3 + angle_rot))
    )
    v1_lattice = r1 + r2
    v2_lattice = r2 + r3
    pixels_pos_true = (
        (center + offset).reshape(2, 1) +
        np.array([v1_lattice, v2_lattice]).transpose().dot(pixels_nvs)
    )
    n_pixels = pixels_nvs.shape[1]
    print(
        'test lattice with v1=', v1_lattice,
        'v2=', v2_lattice,
        'offset=', offset)
    for pixel in range(n_pixels):
        pos_true = pixels_pos_true[:, pixel]
        for i in range(10, -1, -1):
            image = set_hexagon(
                image,
                pos_true,
                r1=(i + 8) / 20 * r1,
                r2=(i + 8) / 20 * r2,
                value=1 - i / 10)
        image = set_hexagon(
            image,
            pos_true,
            r1=7 / 20 * r1,
            r2=7 / 20 * r2,
            value=0)
    image += noise_ampl * np.random.randn(image.shape[0], image.shape[1])
    # add bright pixels so test image can pass the same
    # cleaning procedure as the true images
    image[0:100, 0:100] = 200 * np.ones((100, 100))
    if output_filename is not None:
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = plt.gca()
        plt.imshow(image, cmap='gray', vmin=0, vmax=10)
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(output_filename, 'saved.')
    return image, pixels_pos_true


def get_pixel_nvs(digicam_config_file=camera_config_file):
    """Camera and Geometry objects
    (mapping, pixel, patch + x,y coordinates pixels)
    """
    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)
    pixels_pos_mm = np.array(
        [digicam_geometry.pix_x.to(u.mm), digicam_geometry.pix_y.to(u.mm)]
    ).transpose()
    pixels_pos_mm = pixels_pos_mm.dot(np.array([[0, -1], [1, 0]]))
    pixels_v1 = pixels_pos_mm[
        digicam_geometry.neighbors[0][1], :] - pixels_pos_mm[0, :]
    pixels_v2 = pixels_pos_mm[
        digicam_geometry.neighbors[0][0], :] - pixels_pos_mm[0, :]
    index_to_pos = np.array([pixels_v1, pixels_v2]).transpose()
    relative_pos = (pixels_pos_mm - pixels_pos_mm[0, :]).transpose()
    pixels_nvs = np.linalg.pinv(index_to_pos).dot(relative_pos)
    pixels_nvs -= np.mean(pixels_nvs, axis=1).reshape(2, 1)
    return pixels_nvs
