from digicampipe.utils import geometry
from digicampipe.image.kernels import *
from digicampipe.image.utils import *
from cts_core.camera import Camera
from astropy import units as u
from matplotlib.patches import Circle, Arrow
import numpy as np
import os


class ConesImage(object):
    def __init__(self, image, image_cone=None, output_dir=None, digicam_config_file='./resources/camera_config.cfg'):
        """
        constructor of a ConesImage object.
        :param image: fit filename or numpy array containing the lid CCD image.
        If set to 'test', a test image is created.
        :param image_cone: optional fit filename of the cone image. the fit file is created calling get_cone()
        :param output_dir: optional directory where to put the original lid CCD image in case of test
        """
        self.filename = None
        # Camera and Geometry objects (mapping, pixel, patch + x,y coordinates pixels)
        digicam = Camera(_config_file=digicam_config_file)
        digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)
        pixels_pos_mm = np.array([digicam_geometry.pix_x.to(u.mm), digicam_geometry.pix_y.to(u.mm)]).transpose()
        pixels_pos_mm = pixels_pos_mm.dot(np.array([[0, -1], [1, 0]]))
        pixels_v1 = pixels_pos_mm[digicam_geometry.neighbors[0][1], :] - pixels_pos_mm[0, :]
        pixels_v2 = pixels_pos_mm[digicam_geometry.neighbors[0][0], :] - pixels_pos_mm[0, :]
        # plot_points(self.pixels_pos_mm[:, 0], self.pixels_pos_mm[:, 1])
        index_to_pos = np.array([pixels_v1, pixels_v2]).transpose()
        relative_pos = (pixels_pos_mm - pixels_pos_mm[0, :]).transpose()
        self.pixels_nvs = np.linalg.pinv(index_to_pos).dot(relative_pos)
        self.pixels_nvs -= np.round(np.mean(self.pixels_nvs, axis=1)).reshape(2, 1)
        self.pixels_nvs = np.round(self.pixels_nvs).astype(int)
        if type(image) is str:
            if image == 'test':
                angle_rot = 3.5 / 180 * np.pi
                offset = np.array((25, 5))
                image = np.zeros((2472, 3296))
                center = (np.array(image.shape[::-1]) - 1) / 2
                r1 = 38.3 * np.array((np.cos(angle_rot), np.sin(angle_rot)))
                r2 = 38.3 * np.array((np.cos(np.pi / 3 + angle_rot), np.sin(np.pi / 3 + angle_rot)))
                v1_lattice = 2 * r1 - r2
                v2_lattice = r1 + r2
                print('test lattice with v1=', v1_lattice, 'v2=', v2_lattice, 'offset=', offset)
                for nv1, nv2 in self.pixels_nvs.transpose():
                    for i in range(10, -1, -1):
                        image = set_hexagon(image, center + offset + nv1 * v1_lattice + nv2 * v2_lattice,
                                            r1=(i+7)/20 * r1, r2=(i+7)/20 * r2, value=1-i/10)
                    image = set_hexagon(image, center + offset + nv1 * v1_lattice + nv2 * v2_lattice,
                                        r1=6/20 * r1, r2=6/20 * r2, value=0)
                image += 0. * np.random.randn(image.shape[0], image.shape[1])
                image[0, 0] = 10000
                if output_dir is not None:
                    fig = plt.figure()
                    ax = plt.gca()
                    plt.imshow(image, cmap='gray', vmin=0, vmax=10)
                    plt.grid(None)
                    plt.axis('off')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    output_filename = os.path.join(output_dir, 'cones-original.png')
                    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    print(output_filename, 'saved.')
            else:
                self.filename = image
                image = fits.open(image)[0].data
        if type(image) is not np.ndarray:
            raise AttributeError('image must be a filename or a numpy.ndarray')
        # high pass filter
        self.image_cones = signal.convolve2d(image, high_pass_filter_77, mode='same', boundary='symm')
        # low_pass_filter
        self.image_cones = signal.convolve2d(self.image_cones, gauss(3, (15, 15)), mode='same', boundary='symm')
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
        self.v2_lattice = None  # distance between 2 hexagons
        self.r1 = None  # radius of the hexagon pixel
        self.r2 = None  # radius of the hexagon (after r1 in anticlockwise)
        self.r3 = None  # radius of the hexagon (after r2 in anticlockwise)
        # individual cone image, call get_cone() function to compute it
        if image_cone is None:
            self.image_cone = None
        else:
            hdu = fits.open(image_cone)[0]
            self.image_cone = hdu.data
            self.center_fitted = np.array((np.real(hdu.header['center']), np.imag(hdu.header['center'])))
            center = (np.array(self.image_cones.shape[::-1]) - 1) / 2
            print('center of loaded cone at', self.center_fitted, ',', self.center_fitted - center, 'from center')
            ks1 = np.array((np.real(hdu.header['ks1']), np.imag(hdu.header['ks1'])))
            ks2 = np.array((np.real(hdu.header['ks2']), np.imag(hdu.header['ks2'])))
            self.ks = np.array((ks1, ks2))
            self.v1_lattice = np.array((np.real(hdu.header['v1']), np.imag(hdu.header['v1'])))
            self.v2_lattice = np.array((np.real(hdu.header['v2']), np.imag(hdu.header['v2'])))
            self.r1 = np.array((np.real(hdu.header['r1']), np.imag(hdu.header['r1'])))
            self.r2 = np.array((np.real(hdu.header['r2']), np.imag(hdu.header['r2'])))
            self.r3 = np.array((np.real(hdu.header['r3']), np.imag(hdu.header['r3'])))

    def plot_cones(self, radius_mask=None, output_dir=None):
        """
        If radius_mask is None, plot the lid CCD image after filtering, otherwise plot the cones contents of the image.
        The cones contents is obtained applying a mask of radius radius_mask around each peak of the FFT and doing the
        inverse FFT transformation.
        :param radius_mask: optional radius of the mask used for each peak in the FFT
        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed
        instead of being saved to a file.
        """
        if output_dir is not None:
            plt.ioff()
        fig = plt.figure()
        ax = plt.gca()
        if type(self.image_cones) is not np.ndarray:
            raise AttributeError([self.filename, ' must be a fit file'])
        if radius_mask is not None:
            mask = self.get_fft_mask(radius=radius_mask)
            image_cones = np.real(np.fft.ifft2(self.fft_image_cones * mask))
            image_cones -= np.min(image_cones)
        else:
            image_cones = self.image_cones
        plt.imshow(image_cones, cmap='gray')
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
        if output_dir is None:
            plt.show()
        else:
            if self.filename is not None:
                if radius_mask is not None:
                    output_filename = self.filename.replace('.fits', '-cones-filtered.png')
                else:
                    output_filename = self.filename.replace('.fits', '-cones.png')
            else:
                if radius_mask is not None:
                    output_filename = os.path.join(output_dir, 'cones-filtered.png')
                else:
                    output_filename = os.path.join(output_dir, 'cones.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')

    def scan_cone_position(self, radius_mask, output_dir=None, center_scan=None, rotations=(60, 300)):
        """
        Calculate the hexagonalicity for each pixel inside a camera's pixel.
        :param radius_mask: radius of the mask used for extracting pixels from image (see plot_cones())
        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed
        instead of being saved to a file.
        :param center_scan: optional position of the center of the camera's pixel.
        Center of image is used if None (default)
        :param rotations: angles in degrees used for calculating hexagonalicity.

        remark :  output should look like
        http://n.ethz.ch/~nielssi/download/4.%20Semester/AC%20II/Unterlagen/symmetry_2D_3.pdf page 22
        """
        if type(radius_mask) is not float:
            raise AttributeError('radius_mask must be a float.')
        if self.r1 is None or self.r2 is None:
            raise AttributeError('camera pixel geometry must be computed prior of calling scan_cone_position().')
        mask = self.get_fft_mask(radius=radius_mask)
        image_cones = np.real(np.fft.ifft2(self.fft_image_cones * mask))
        center_image = (np.array(image_cones.shape[::-1]) - 1) / 2
        if center_scan is None:
            center_scan = center_image
        scan_area = np.zeros(image_cones.shape, dtype=bool)
        scan_area = set_hexagon(scan_area, center_scan, r1=self.r1, r2=self.r2, value=1)
        scan_result = np.zeros(image_cones.shape, dtype=float)
        all_pixels_y, all_pixels_x = np.indices(scan_area.shape)
        pixels_x = all_pixels_x[scan_area == 1].flatten()
        pixels_y = all_pixels_y[scan_area == 1].flatten()
        npixel = pixels_x.shape[0]
        print('calculating hexagonalicity for each position in the pixel:')
        last_precent = 0
        for pixel_x, pixel_y, iter in zip(pixels_x[::1], pixels_y[::1], range(npixel)):
            percent_done = np.floor((iter+1)*100/npixel)
            if percent_done > last_precent:
                print(percent_done, '%')
                last_precent = percent_done
            hex = -get_neg_hexagonalicity_with_mask((pixel_x, pixel_y), image_cones, self.r1, self.r2,
                                                    rotations=rotations)
            scan_result[pixel_y, pixel_x] = hex
        if output_dir is not None:
            plt.ioff()
        else:
            plt.ion()
        fig = plt.figure()
        ax = plt.subplot(1, 2, 2)
        vmin = np.min(scan_result[scan_result > 0])
        vmax = np.max(scan_result[scan_result > 0])
        plt.imshow(scan_result, cmap='gray', vmin=vmin, vmax=vmax)
        max_y, max_x = np.unravel_index(np.argmax(scan_result), dims=scan_result.shape)
        plt.plot(max_x, max_y, 'r+')
        plt.xlim((np.min(pixels_x), np.max(pixels_x)))
        plt.ylim((np.min(pixels_y), np.max(pixels_y)))
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(1, 2, 1)
        plt.imshow(image_cones * scan_area, cmap='gray')
        plt.plot(max_x, max_y, 'r+')
        plt.xlim((np.min(pixels_x), np.max(pixels_x)))
        plt.ylim((np.min(pixels_y), np.max(pixels_y)))
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if output_dir is None:
            plt.show()
        else:
            if self.filename is not None:
                output_filename = self.filename.replace('.fits', '-hexagonalicity.png')
            else:
                output_filename = os.path.join(output_dir, 'hexagonalicity.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')

    def get_cone(self, radius_mask, output_dir=None, save_to_file=True):
        """
        function to find the center of a camera's pixel
        :param radius_mask: radius of the mask used for extracting pixels from image (see plot_cones())
        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed
        instead of being saved to a file.
        :param save_to_file: boolean (default = True), should the resulting image be saved to a fit files along
        with important parameters ?
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
        offsets= (0*self.r1, (self.r1+self.r2)/3, self.r1, self.r2, (self.r2+self.r3)/3, self.r3,
                  (self.r3-self.r1)/3, -self.r1, (-self.r1-self.r2)/3, -self.r2,
                  (-self.r2-self.r3)/3, -self.r3, (-self.r3+self.r1)/3)
        results = []
        for i, offset in enumerate(offsets):
            res = optimize.minimize(get_neg_hexagonalicity_with_mask,
                                    center_tested+offset, args=(image_cones, self.r1, self.r2, (60, 300)),
                                    bounds=bounds, method='TNC',
                                    options={'disp': False, 'eps': 1, 'xtol': 1e-2, 'maxiter': 200})
            results.append(res)
            print('fit', i+1, '/', len(offsets))
            if -res.fun > 0.5:
                break
        hex_results = np.array([-res.fun for res in results])
        pos_results = np.array([res.x for res in results])
        self.center_fitted = pos_results[np.argmax(hex_results)]
        # fine fit for best result
        res = optimize.minimize(get_neg_hexagonalicity_with_mask,
                                self.center_fitted, args=(image_cones, self.r1, self.r2, (60, 120, 180, 240, 300)),
                                bounds=bounds, method='TNC',
                                options={'disp': False, 'eps': .01, 'xtol': 1e-4, 'maxiter': 200})
        results.append(res)
        hex_results = np.array([-res.fun for res in results])
        pos_results = np.array([res.x for res in results])
        self.center_fitted = pos_results[np.argmax(hex_results)]
        hex_result = hex_results[np.argmax(hex_results)]
        print('pixel center found: ', self.center_fitted, 'with hex=', hex_result)
        print(self.center_fitted - center_image, 'px from center')
        points_fitted = points - center_tested + np.array(self.center_fitted)
        pixels_x_min = int(np.floor(np.min(points_fitted[:, 0])))
        pixels_x_max = int(np.ceil(np.max(points_fitted[:, 0]))) + 1
        pixels_y_min = int(np.floor(np.min(points_fitted[:, 1])))
        pixels_y_max = int(np.ceil(np.max(points_fitted[:, 1]))) + 1
        # plot_image(image_cones)
        image_crop = image_cones[pixels_y_min:pixels_y_max, pixels_x_min:pixels_x_max]
        # plot_image(image_crop)
        center_crop = np.array(self.center_fitted) - np.array((pixels_x_min, pixels_y_min))
        mask_hexa = np.zeros_like(image_crop)
        mask_hexa = set_hexagon(mask_hexa, center=center_crop, r1=self.r1, r2=self.r2)
        self.image_cone = image_crop * mask_hexa
        if save_to_file:
            hdu = fits.PrimaryHDU(self.image_cone)
            hdu.header['center'] = (self.center_fitted[0] + 1j * self.center_fitted[1],
                                    'fitted position (in original image) of the hexagon center')
            hdu.header['ks1'] = (self.ks[0, 0] + 1j * self.ks[0, 1],
                                 '1st vector of the base of the hexagonal lattice in reciprocal space')
            hdu.header['ks2'] = (self.ks[1, 0] + 1j * self.ks[1, 1],
                                 '2nd vector of the base of the hexagonal lattice in reciprocal space')
            hdu.header['v1'] = (self.v1_lattice[0] + 1j * self.v1_lattice[1],
                                'spacing between 2 hexagons along the 1st axis')
            hdu.header['v2'] = (self.v2_lattice[0] + 1j * self.v2_lattice[1],
                                'spacing between 2 hexagons along the 2nd axis')
            hdu.header['r1'] = (self.r1[0] + 1j * self.r1[1], '1st radius of the hexagon')
            hdu.header['r2'] = (self.r2[0] + 1j * self.r2[1], '2nd radius of the hexagon')
            hdu.header['r3'] = (self.r3[0] + 1j * self.r3[1], '3rd radius of the hexagon')
            if self.filename is not None:
                cone_filename = self.filename.replace('.fits', '-cone.fits')
            else:
                cone_filename = os.path.join(output_dir, 'cone.fits')
            hdu.writeto(cone_filename, overwrite=True)
            print('cone saved to ', cone_filename)
        if output_dir is not None:
            plt.ioff()
        else:
            plt.ion()
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(self.image_cone, cmap='gray')
        plt.plot(self.center_fitted[0]-pixels_x_min, self.center_fitted[1]-pixels_y_min, 'y+')
        plt.xlim((-0.5, pixels_x_max-pixels_x_min-0.5))
        plt.ylim((-0.5, pixels_y_max-pixels_y_min-0.5))
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if output_dir is None:
            plt.show()
        else:
            if self.filename is not None:
                output_filename = self.filename.replace('.fits', '-cone.png')
            else:
                output_filename = os.path.join(output_dir, 'cone.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')

    def plot_fft_cones(self, radius_mask=None, output_dir=None):
        """
        plot the FFT of the filtered lid CCD image. If radius_mask is given, everything more distant than it from
        a peak is set to 0.
        :param radius_mask: radius in pixels of circular mask around each peak. Default = no mask.
        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed
        instead of being saved to a file.
        """
        if output_dir is not None:
            plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if radius_mask is not None:
            mask = self.get_fft_mask(radius=radius_mask)
            fft_image_cones = self.fft_image_cones * mask
        else:
            fft_image_cones = self.fft_image_cones
        plt.imshow(np.abs(np.fft.fftshift(fft_image_cones)), cmap='gray')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if output_dir is None:
            plt.show()
        else:
            if self.filename is not None:
                if radius_mask is not None:
                    output_filename = self.filename.replace('.fits', '-cones-fft-masked.png')
                else:
                    output_filename = self.filename.replace('.fits', '-cones-fft.png')
            else:
                if radius_mask is not None:
                    output_filename = os.path.join(output_dir, 'cones-fft-masked.png')
                else:
                    output_filename = os.path.join(output_dir, 'cones-fft.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')

    def get_fft_mask(self, radius=3):
        """
        function creating a mask to keep only pixels closer to peaks than radius.
        :param radius: radius in pixels of circular mask around each peak.
        :return: the mask to be applied on fft (image of same size as self.fft_image_cones).
        """
        if self.ks is None:
            self.get_cones_separation_reciprocal()
        center = (np.array((self.fft_image_cones.shape[::-1])) - 1) / 2
        mask = make_repetitive_mask(self.fft_image_cones.shape,
                                    radius, self.ks[0, :], self.ks[1, :], center, nrepetition=100)
        for nk1 in range(-50, 50):
            for nk2 in range(-50, 50):
                center_circle = center + nk1 * self.ks[0, :] + nk2 * self.ks[1, :]
                mask = set_circle(mask, center=center_circle.reshape((2,)), radius=radius, value=1)
        return np.fft.ifftshift(mask)

    def get_cones_separation_reciprocal(self, output_dir=None):
        """
        Function to calculate the distance between 2 neighbour pixels (as vectors) and 3 consecutive radius.
        The distances are stored in self.v[12]_lattice and the radius are stored in self.r[123]
        :param output_dir: optional directory where to put the auto-correlation of the FFT image.
        """
        fft_image_cones_shifted = signal.convolve2d(np.abs(self.fft_image_cones), high_pass_filter_77,
                                                    mode='same', boundary='wrap')
        fft_image_cones_shifted = np.fft.fftshift(fft_image_cones_shifted)
        fft_image_cones_shifted[fft_image_cones_shifted < 0] = 0
        ks_base, auto_correlation_saved, center_peaks = get_peaks_separation(fft_image_cones_shifted, crop_range=802)
        if output_dir is not None:
            plt.ioff()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(auto_correlation_saved, cmap='gray')
            plt.plot(center_peaks[:, 0], center_peaks[:, 1], 'y+')
            plt.xlim((0, auto_correlation_saved.shape[1]))
            plt.ylim((0, auto_correlation_saved.shape[0]))
            k1_points = np.array((np.zeros((2,)), ks_base[0, :])) + center_peaks[0, :]
            k2_points = np.array((np.zeros((2,)), ks_base[1, :])) + center_peaks[0, :]
            plt.plot(k1_points[:, 0], k1_points[:, 1], 'b-', linewidth=1)
            plt.plot(k2_points[:, 0], k2_points[:, 1], 'r-', linewidth=1)
            plt.grid(None)
            if self.filename is not None:
                output_filename = self.filename.replace('.fits', '-cones-sep-reciprocal.png')
            else:
                output_filename = os.path.join(output_dir, 'cones-sep-reciprocal.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')
        self.ks = ks_base
        self.v1_lattice, self.v2_lattice = reciprocal_to_lattice_space(self.ks[0, :], self.ks[1, :], self.fft_image_cones.shape[::-1])
        r3 = (self.v1_lattice + self.v2_lattice) / 3
        r1 = self.v1_lattice - r3
        r2 = self.v2_lattice - r3
        self.r1, self.r2 = get_consecutive_hex_radius(r1, r2)
        self.r3 = self.r2 - self.r1
        print("r1=", self.r1, "|r1|=", np.abs(self.r1[0] + 1j * self.r1[1]))
        print("r2=", self.r2, "|r2|=", np.abs(self.r2[0] + 1j * self.r2[1]))
        print("r3=", self.r3, "|r3|=", np.abs(self.r3[0] + 1j * self.r3[1]))

    def plot_cones_presence(self, radius_mask=None, output_dir=None):
        """
        Do convolution of cone image on filtered lid CCD image, then fit peaks positions.
        Find the best orientation and position for the camera geometry to match the peaks.
        Finally refit the distance between pixels only taking into account the matching peaks.
        :param radius_mask: radius of the mask used to keep only relevant peaks in the convultion.
        :param output_dir: optional directory where to put the resulting images.
        If None (default) the image is displayed instead of being saved to a file.
        """
        if self.image_cone is None:
            if radius_mask is None:
                raise AttributeError("cone_image was not calculated and no radius mask given")
            self.get_cone(radius_mask=radius_mask, output_dir=output_dir)
        print('find peaks')
        cone_presence = signal.fftconvolve(self.image_cones, self.image_cone, mode='same')
        cone_presence = signal.fftconvolve(cone_presence, high_pass_filter_2525, mode='same')
        cone_presence[cone_presence < 0] = 0
        if output_dir is not None:
            plt.ioff()
        else:
            plt.ion()
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(cone_presence, cmap='gray')
        plt.grid(None)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if output_dir is None:
            plt.show()
        else:
            if self.filename is not None:
                output_filename = self.filename.replace('.fits', '-cones-presence.png')
            else:
                output_filename = os.path.join(output_dir, 'cones-presence.png')
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(output_filename, 'saved.')
        if radius_mask is not None:
            print('find distance between pixels')
            cone_presence_fft_shifted = np.fft.fftshift(np.abs(np.fft.fft2(cone_presence)))
            cone_presence_fft_shifted = signal.fftconvolve(cone_presence_fft_shifted, high_pass_filter_77, mode='same')
            ks, auto_correlation, center_peaks = get_peaks_separation(cone_presence_fft_shifted,
                                                                      center=None, crop_range=800, radius_removed=20)
            del cone_presence_fft_shifted
            if output_dir is not None:
                plt.ioff()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.imshow(auto_correlation, cmap='gray')
                plt.plot(center_peaks[:, 0], center_peaks[:, 1], 'y+')
                plt.xlim((0, auto_correlation.shape[1]))
                plt.ylim((0, auto_correlation.shape[0]))
                k1_points = np.array((0 * ks[0, :], ks[0, :])) + center_peaks[0, :]
                k2_points = np.array((0 * ks[0, :], ks[1, :])) + center_peaks[0, :]
                plt.plot(k1_points[:, 0], k1_points[:, 1], 'b-', linewidth=1)
                plt.plot(k2_points[:, 0], k2_points[:, 1], 'r-', linewidth=1)
                plt.grid(None)
                if self.filename is not None:
                    output_filename = self.filename.replace('.fits', '-cones-sep.png')
                else:
                    output_filename = os.path.join(output_dir, 'cones-sep.png')
                plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                print(output_filename, 'saved.')
            v1, v2 = reciprocal_to_lattice_space(ks[0, :],  ks[1, :], cone_presence.shape[::-1])
            v1, v2 = get_consecutive_hex_radius(v1, v2)
            v3 = v2 - v1
            v1_length = np.abs(v1[0] + 1j * v1[1])
            v2_length = np.abs(v2[0] + 1j * v2[1])
            v3_length = np.abs(v3[0] + 1j * v3[1])
            print("v1=", v1, "|v1|=", v1_length)
            print("v2=", v2, "|v2|=", v2_length)
            print("v3=", v3, "|v3|=", v3_length)
            pixels_fit_px = []
            pixels_fit_sigma = []
            nfail = 0
            print('fit all peaks')
            for i1 in range(-100, 100):
                for i2 in range(-100, 100):
                    peak_pos_aprox = self.center_fitted + i1 * v1 + i2 * v2
                    crop_px1 = np.floor(peak_pos_aprox - radius_mask)
                    crop_px1 = np.maximum(crop_px1, (0, 0))
                    crop_px1 = np.minimum(crop_px1, (cone_presence.shape[1] - 1, cone_presence.shape[0] - 1))
                    crop_px1 = crop_px1.astype(int)
                    crop_px2 = np.ceil(peak_pos_aprox + radius_mask)
                    crop_px2 = np.maximum(crop_px2, (0, 0))
                    crop_px2 = np.minimum(crop_px2, (cone_presence.shape[1] - 1, cone_presence.shape[0] - 1))
                    crop_px2 = crop_px2.astype(int)
                    if np.any(crop_px2 - crop_px1 <= np.round(radius_mask)):
                        continue
                    crop_center = (crop_px2 - crop_px1 - 1.) / 2
                    # print('fit around:', peak_pos_aprox)
                    peak_crop, crop_px1, crop_px2 = crop_image(cone_presence, crop_px1, crop_px2)
                    max_pos_crop = np.argmax(peak_crop)
                    [max_pos_y, max_pos_x] = np.unravel_index(max_pos_crop, peak_crop.shape)
                    init_amplitude = peak_crop[max_pos_y, max_pos_x] - np.min(peak_crop)
                    init_param = (init_amplitude, max_pos_x, max_pos_y, 4, 4, 0, np.min(peak_crop))
                    fit_result, success = FitGauss2D(peak_crop.transpose(), ip=init_param)
                    amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg = fit_result
                    if 0 < success <= 4 and 0 < xsigma < 6 and 0 < ysigma < 6 and\
                                            -5 < xcenter - crop_center[0] < 5 and -5 < ycenter - crop_center[1] < 5:
                        # print('SUCCESS: xcenter:', np.round(xcenter + crop_px1[0], 2),
                        #       'ycenter:', np.round(ycenter + crop_px1[0], 2),
                        #       'xsigma:', np.round(xsigma,2), 'ysigma:', np.round(ysigma,2))
                        pixels_fit_px.append(np.array([xcenter, ycenter]) + crop_px1)
                        pixels_fit_sigma.append(np.array([xsigma, ysigma]))
                    else:
                        # print('FAIL: xcenter:', np.round(xcenter, 2), 'ycenter:', np.round(ycenter, 2),
                        #       'xsigma:', np.round(xsigma, 2), 'ysigma:', np.round(ysigma, 2))
                        nfail += 1
                    if np.mod(len(pixels_fit_px)+nfail, 100) == 0:
                        print(len(pixels_fit_px)+nfail, 'fits done')
            pixels_fit_px = np.array(pixels_fit_px)
            pixels_fit_sigma = np.array(pixels_fit_sigma)
            print('success =', pixels_fit_px.shape[0], '/', pixels_fit_px.shape[0] + nfail)
            print('look for pixels geometry:')
            best_match_nv1, best_match_nv2 = 0, 0
            best_match_v1, best_match_v2 = v1, v2
            best_matching_nvs = 0
            for v1_test, v2_test in [[v1, v2], [v2, v3], [v3, -v1], [-v1, -v2], [-v2, -v3], [-v3, v1]]:
                v_matrix = np.array([v1_test, v2_test]).transpose()
                pixels_fit_nvs = np.round(np.linalg.pinv(v_matrix).dot((pixels_fit_px - self.center_fitted).transpose()))
                pixels_fit_nvs_set = set(map(tuple, pixels_fit_nvs.transpose().astype(int)))
                for nv1 in range(-10, 10):
                    for nv2 in range(-10, 10):
                        pixels_nvs = map(tuple, self.pixels_nvs.transpose() + np.array([nv1, nv2]))
                        matching_nvs = len(pixels_fit_nvs_set.intersection(pixels_nvs))
                        if matching_nvs > best_matching_nvs:
                            best_matching_nvs = matching_nvs
                            best_match_nv1, best_match_nv2 = nv1, nv2
                            best_match_v1, best_match_v2 = v1_test, v2_test
            v1, v2 = best_match_v1, best_match_v2
            print('best base: v1=', v1, ', v2=', v2)
            print('best match: nv1 =', best_match_nv1, ', nv2 =', best_match_nv2)
            v_matrix = np.array([v1, v2]).transpose()
            self.center_fitted += best_match_nv1 * v1 + best_match_nv2 * v2
            pixels_fit_nvs = np.round(np.linalg.pinv(v_matrix).dot((pixels_fit_px - self.center_fitted).transpose()))
            pixels_fit_nvs_set = set(map(tuple, pixels_fit_nvs.transpose().astype(int)))
            nvs_fit_matching_set = pixels_fit_nvs_set.intersection(map(tuple, self.pixels_nvs.transpose()))
            is_fit_matching = np.array([x in nvs_fit_matching_set for x in map(tuple, pixels_fit_nvs.transpose())])
            fit_not_matching = is_fit_matching == 0
            is_pixel_fitted = np.array([x in nvs_fit_matching_set for x in map(tuple, self.pixels_nvs.transpose())])
            pixels_not_fitted = is_pixel_fitted == 0
            print(np.sum(is_fit_matching), 'fits in best match,', np.sum(fit_not_matching), 'outside')
            print('global fit of lattice base vectors, center was ', self.center_fitted)
            n_matching = np.sum(is_fit_matching)
            nvs = np.vstack((pixels_fit_nvs[:, is_fit_matching], np.ones((1, n_matching))))
            pxs = pixels_fit_px[is_fit_matching, :].transpose()
            precise_vs = pxs.dot(np.linalg.pinv(nvs))
            self.v1_lattice = precise_vs[:, 0]
            self.v2_lattice = precise_vs[:, 1]
            self.center_fitted = precise_vs[:, 2]
            pixels_geom_px = precise_vs.dot(np.vstack((self.pixels_nvs,  np.ones((1, self.pixels_nvs.shape[1])))))
            print("v1=", self.v1_lattice, "|v1|=", np.abs(self.v1_lattice[0] + 1j * self.v1_lattice[1]))
            print("v2=", self.v2_lattice, "|v2|=", np.abs(self.v2_lattice[0] + 1j * self.v2_lattice[1]))
            center_image = (np.array(self.image_cones.shape[::-1]) - 1) / 2
            print("center=", self.center_fitted, ',', self.center_fitted - center_image, 'from center')
            fig = plt.figure()
            ax = plt.gca()
            plt.imshow(cone_presence, cmap='gray')
            plt.autoscale(False)
            for center_pixel in pixels_geom_px[:, is_pixel_fitted].transpose():
                circle = Circle((center_pixel[0], center_pixel[1]), radius=radius_mask + 10, fill=False, color='g')
                ax.add_artist(circle)
            for center_pixel in pixels_geom_px[:, pixels_not_fitted].transpose():
                circle = Circle((center_pixel[0], center_pixel[1]), radius=radius_mask + 10, fill=False, color='r')
                ax.add_artist(circle)
            plt.errorbar(pixels_fit_px[is_fit_matching, 0], pixels_fit_px[is_fit_matching, 1],
                         xerr=pixels_fit_sigma[is_fit_matching, 0] * 3, yerr=pixels_fit_sigma[is_fit_matching, 1] * 3,
                         fmt='b', linestyle='none', elinewidth=1)
            plt.errorbar(pixels_fit_px[fit_not_matching, 0], pixels_fit_px[fit_not_matching, 1],
                         xerr=pixels_fit_sigma[fit_not_matching, 0] * 3, yerr=pixels_fit_sigma[fit_not_matching, 1] * 3,
                         fmt='y', linestyle='none', elinewidth=1)
            plt.grid(None)
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            pixels_px_prediction = precise_vs.dot(nvs)
            residuals = pxs - pixels_px_prediction
            good_residuals = np.abs(residuals[0, :] + 1j * residuals[1, :]) < 2.
            bad_residuals = good_residuals == 0
            print(np.sum(good_residuals), "residuals below threshold,", np.sum(bad_residuals), "above.")
            print('mean(residuals)=', np.mean(residuals[:, good_residuals], axis=1),
                  'std(residual)=', np.std(residuals[:, good_residuals], axis=1))
            if output_dir is None:
                plt.show()
            else:
                if self.filename is not None:
                    output_filename = self.filename.replace('.fits', '-cones-presence-filtered.png')
                else:
                    output_filename = os.path.join(output_dir, 'cones-presence-filtered.png')
                plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                print(output_filename, 'saved.')
            fig = plt.figure()
            ax = plt.gca()
            plt.imshow(self.image_cones, cmap='gray')
            plt.autoscale(False)
            ploted_residuals = residuals[:, good_residuals].transpose()
            ploted_prediction = pixels_px_prediction[:, good_residuals].transpose()
            for residual, center_pixel in zip(ploted_residuals, ploted_prediction):
                arrow = Arrow(center_pixel[0], center_pixel[1], residual[0]*10, residual[1]*10,
                              color='b', width=15., linewidth=1)
                ax.add_artist(arrow)
            ploted_residuals = residuals[:, bad_residuals].transpose()
            ploted_prediction = pixels_px_prediction[:, bad_residuals].transpose()
            for residual, center_pixel in zip(ploted_residuals, ploted_prediction):
                arrow = Arrow(center_pixel[0], center_pixel[1], residual[0]*10, residual[1]*10,
                              color='r', width=15., linewidth=1)
                ax.add_artist(arrow)
            plt.grid(None)
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if output_dir is None:
                plt.show()
            else:
                if self.filename is not None:
                    output_filename = self.filename.replace('.fits', '-cones-fit-residuals.png')
                else:
                    output_filename = os.path.join(output_dir, 'cones-fit-residuals.png')
                plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                print(output_filename, 'saved.')
