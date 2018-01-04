from astropy.stats import sigma_clipped_stats
import numpy as np
import scipy
from skimage.draw import polygon
import matplotlib.pyplot as plt
import cv2
from scipy import signal


from photutils import DAOStarFinder


class Rectangle:
    def __init__(self, left, bottom, right, top):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        if left > right:
            self.left, self.right = right, left
        else:
            self.left, self.right = left, right
        if bottom > top:
            self.bottom, self.top = top, bottom
        else:
            self.bottom, self.top = bottom, top

    def __str__(self):
        return "Rectangle(left=, bottom=, right=, top=)" % (
            self.left, self.bottom, self.right, self.top)

    def width(self):
        return self.right - self.left

    def height(self):
        return self.top - self.bottom

    def center(self):
        center_x = (self.left + self.right) / 2
        center_y = (self.bottom + self.top) / 2
        return (center_x, center_y)


class CroppedImage:
    def __init__(self, image, rectangle, strict_limit=True):
        npixel_y, npixel_x = image.shape
        if strict_limit:
            if not (
                0 <= rectangle.left < npixel_x and
                0 <= rectangle.bottom < npixel_y
            ):
                raise AttributeError(
                    "invalid crop_pixel position:", rectangle,
                    "image size is: ", image.shape)
            if not (
                0 < rectangle.right < npixel_x and
                0 < rectangle.top < npixel_y
            ):
                raise AttributeError(
                    "invalid crop_pixel position:", rectangle,
                    "image size is: ", image.shape)
        else:
            rectangle.left = max((0, rectangle.left))
            rectangle.bottom = max((0, rectangle.bottom))
            rectangle.right = min((npixel_x - 1, rectangle.right))
            rectangle.top = min((npixel_y - 1, rectangle.top))
            if (
                rectangle.left == rectangle.right or
                rectangle.bottom == rectangle.top
            ):
                raise AttributeError(
                    "empty crop region:", rectangle,
                    "image size is: ", image.shape
                )
        self.image = image[
            rectangle.bottom:rectangle.top,
            rectangle.left:rectangle.right
        ]
        self.rectangle = rectangle


def crop_image(image, crop_pixel1, crop_pixel2):
    """return sub-image out of image

    Parameters
    ----------
    image : 2d-array
        image data with a shape = (height, width)
    crop_pixel1 : iterable of 2 ints
        position (pos_x , pos_y) of one corner of the returned region
    crop_pixel2 : iterable of 2 ints
        position (pos_x , pos_y) of the opposite corner of the returned region

    Returns
    -------
    tuple of (image, 1st corner, 2nd corner)
        the corners reflect the actual region used for cropping,
        ignoring areas outside the available image
    """
    if type(image) is str:
        image = fits.open(image)[0].data
    if type(image) is not np.ndarray:
        raise AttributeError("image must be a filename or a numpy array.")
    if crop_pixel1 is not list and len(crop_pixel1) is not 2:
        raise AttributeError(
            "crop_pixel1 must be a list of 2 ints",
            "(position in px of one corner of the crop area).",
        )
    if crop_pixel2 is not list and len(crop_pixel1) is not 2:
        raise AttributeError(
            "crop_pixel2 must be a list of 2 ints",
            "(position in px of the other corner of the crop area).",
        )
    crop_xmin, crop_ymin = crop_pixel1
    crop_xmax, crop_ymax = crop_pixel2
    if (
        not isinstance(crop_xmin, numbers.Integral) or
        not isinstance(crop_ymin, numbers.Integral)
    ):
        print('Warning in crop_image(): crop_pixel1 does not contain ints')
        crop_xmin = int(crop_xmin)
        crop_ymin = int(crop_ymin)
    if (
        not isinstance(crop_xmax, numbers.Integral) or
        not isinstance(crop_ymax, numbers.Integral)
    ):
        print('Warning in crop_image(): crop_pixel2 does not contain ints')
        crop_xmax = int(crop_xmax)
        crop_ymax = int(crop_ymax)
    npixel_y, npixel_x = image.shape
    if not (0 <= crop_xmin < npixel_x and 0 <= crop_ymin < npixel_y):
        raise AttributeError(
            "invalid crop_pixel position:",
            crop_pixel1,
            crop_pixel2,
            "image size is: ",
            image.shape
        )
    if not (0 < crop_xmax < npixel_x and 0 < crop_ymax < npixel_y):
        raise AttributeError(
            "invalid crop_pixel position:",
            crop_pixel1,
            crop_pixel2,
            "image size is: ",
            image.shape
        )
    if crop_xmin > crop_xmax:
        crop_xmin, crop_xmax = crop_xmax, crop_xmin
    if crop_ymin > crop_ymax:
        crop_ymin, crop_ymax = crop_ymax, crop_ymin
    crop_pixel1 = crop_xmin, crop_ymin
    crop_pixel2 = crop_xmax, crop_ymax
    return (
        image[crop_ymin:crop_ymax, crop_xmin:crop_xmax],
        crop_pixel1,
        crop_pixel2,
    )


def average_images(images):
    # convert to array, if it not already is an ndarray.
    images = np.asarray(images)
    # conversion to 3D array only works, if all images are of equal size.
    assert images.ndim == 3
    return images.mean(axis=-1)


def set_circle(image, center=(0, 0), radius=10, value=1):
    for ix in range(int(center[0] - radius - 1), int(center[0] + radius + 1)):
        for iy in range(
                int(center[1] - radius - 1), int(center[1] + radius + 1)):
            if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]):
                continue  # out of image border
            if np.sqrt(
                (ix - center[0]) * (ix - center[0]) +
                (iy - center[1]) * (iy - center[1])
            ) < radius:
                image[iy, ix] = value
    return image


def make_repetitive_mask(shape, radius, v1, v2, center, nrepetition=100):
    mask = np.zeros(shape)
    for nv1 in range(-int(nrepetition/2), int(nrepetition/2)):
        for nv2 in range(-int(nrepetition/2), int(nrepetition/2)):
            center_circle = center + nv1 * v1 + nv2 * v2
            mask = set_circle(
                mask,
                center=center_circle.reshape((2,)),
                radius=radius,
                value=1
            )
    return mask


def set_parallelogram(image, center=(0, 0), k1=(1, 0), k2=(0, 1), value=1):
    origin = center - k1 / 2 - k2 / 2
    r = np.array(
        (origin[0], origin[0]+k1[0], origin[0]+k1[0]+k2[0], origin[0]+k2[0]))
    c = np.array(
        (origin[1], origin[1]+k1[1], origin[1]+k1[1]+k2[1], origin[1]+k2[1]))
    rr, cc = polygon(r, c, (image.shape[1], image.shape[0]))
    image[cc, rr] = value
    return image


def get_consecutive_hex_radius(r1, r2):
    '''set r2 to be the next radius after r1 anticlockwise
    '''
    r1 = np.array(r1)
    r2 = np.array(r2)
    # set r2 to be the next radius after r1 (anticlockwise)
    angle_r1 = np.angle(r1[0] + 1j * r1[1])
    angle_r2 = np.angle(r2[0] + 1j * r2[1])
    angle_diff = np.mod(angle_r2 - angle_r1, 2 * np.pi)
    resol = 1e-2
    if (
        np.abs(angle_diff) < resol or
        np.abs(angle_diff - np.pi) < resol or
        np.abs(angle_diff) > 2 * np.pi - resol
    ):
        raise AttributeError('r1 and r2 are collinear')
    if np.pi / 2 < angle_diff < np.pi:
        r2 = r1 + r2
    elif np.pi < angle_diff < 3 / 2 * np.pi:
        r2 = -r2
    elif angle_diff > 3 / 2 * np.pi:
        r2 = r1 - r2
    return r1, r2


def set_hexagon(
    image,
    center=(0, 0),
    r1=(1, 0),
    r2=(0.5, np.sqrt(3) / 2),
    value=1,
):
    r1, r2 = get_consecutive_hex_radius(r1, r2)
    # set of hexagon's vertexes
    points = np.array((center + r1,
                       center + r2,
                       center + r2 - r1,
                       center - r1,
                       center - r2,
                       center - r2 + r1))
    rr, cc = polygon(
        points[:, 0],
        points[:, 1],
        (image.shape[1], image.shape[0])
    )
    image[cc, rr] = value
    return image


fig_plot = None


def plot_image(image, wait=True, vmin=None, vmax=None):
    global fig_plot
    plt.ion()
    if fig_plot is None:
        fig_plot = plt.figure()
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry(dx/2, 0, dx/2, dy)
        fig_plot.canvas.flush_events()
    if vmin is None:
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show(block=False)
    fig_plot.canvas.flush_events()
    if wait:
        plt.waitforbuttonpress()


def plot_points(x, y, wait=True):
    global fig_plot
    plt.ion()
    if fig_plot is None:
        fig_plot = plt.figure()
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x0, y0, dx, dy = geom.getRect()
        mngr.window.setGeometry(dx/2, 0, dx/2, dy)
        fig_plot.canvas.flush_events()
    if type(x) is not np.array:
        x = np.array(x)
    if type(y) is not np.array:
        y = np.array(y)
    plt.plot(x, y, '+')
    min_x = np.min(x)
    max_x = np.max(x)
    width_x = max_x - min_x + 1
    plt.xlim((min_x - width_x/10, max_x + width_x/10))
    min_y = np.min(y)
    max_y = np.max(y)
    width_y = max_y - min_y + 1
    plt.xlim((min_y - width_y/10, max_y + width_y/10))
    plt.show(block=False)
    fig_plot.canvas.flush_events()
    if wait:
        plt.waitforbuttonpress()


def get_peaks_separation(
    fft_image_shifted,
    center=None,
    crop_range=None,
    radius_removed=20,
):
    if center is None:
        center = (np.array(fft_image_shifted.shape[::-1]) - 1) / 2
    fft_image_shifted = set_circle(
        fft_image_shifted,
        center=center,
        radius=radius_removed,
        value=0
    )
    if crop_range is None:
        min_crop_x = 0
        max_crop_x = fft_image_shifted.shape[1]
        min_crop_y = 0
        max_crop_y = fft_image_shifted.shape[0]
    else:
        min_crop_x = int(center[1] - crop_range/2)
        max_crop_x = int(center[1] + crop_range/2)
        min_crop_y = int(center[0] - crop_range/2)
        max_crop_y = int(center[0] + crop_range/2)
    crop_fft = fft_image_shifted[min_crop_x:max_crop_x, min_crop_y:max_crop_y]
    # plot_image(crop_fft)
    auto_correlation = signal.fftconvolve(
        crop_fft, crop_fft[::-1, ::-1], mode='same')
    auto_correlation_saved = auto_correlation
    auto_correlation[auto_correlation < 0] = 0
    mean, median, std = sigma_clipped_stats(
        auto_correlation, sigma=3.0, iters=5)
    baseline_sub = auto_correlation - median
    baseline_sub[baseline_sub < 0] = 0
    daofind = DAOStarFinder(fwhm=3.0, threshold=std)
    sources = daofind(baseline_sub)
    center_peaks = np.array(
        [sources['xcentroid'], sources['ycentroid']]).transpose()
    order = np.argsort(sources['mag'])
    center_peaks = center_peaks[order[0:100], :]
    ks = center_peaks[1:, :] - center_peaks[0, :]
    ks_complex = ks[:, 0] + 1j * ks[:, 1]
    angles = np.angle(ks_complex)
    lengths = np.abs(ks_complex)
    short_to_long = np.argsort(lengths)
    ks_base = np.empty((2, 2))
    ks_base[0, :] = ks[short_to_long[0], :]
    angles = angles - np.angle(ks_base[0, 0]+1j*(ks_base[0, 1]))
    factors = np.abs(ks_complex/(ks_base[0, 0]+1j*(ks_base[0, 1])))

    # less than 1 deg from 0
    is_collinear = (np.abs(np.mod(angles+np.pi, 2*np.pi)-np.pi) < 1*np.pi/180)

    # less than 1 deg from pi
    is_anti_collinear = np.abs(np.abs(angles)-np.pi) < 1*np.pi/180
    factors[is_anti_collinear] *= -1
    is_multiple = np.abs(np.mod(factors + 0.5, 1) - 0.5) < 1e-2
    is_used = is_multiple & (is_collinear | is_anti_collinear)
    rounded_factors = np.round(factors[is_used]).reshape(-1, 1)
    ks1s_rescaled = 1 / rounded_factors * ks[is_used, :]
    ks_base[0, :] = np.mean(ks1s_rescaled, axis=0)
    for i in short_to_long[1:]:
        if is_collinear[i] or is_anti_collinear[i]:
            continue  # do not take collinear vector as base
        ks_base[1, :] = ks[i]
        angles = np.angle(ks_complex / (ks_base[1, 0] + 1j * (ks_base[1, 1])))
        factors = np.abs(ks_complex / (ks_base[1, 0] + 1j * (ks_base[1, 1])))

        # less than 1 deg from 0
        is_collinear = (
            np.abs(
                np.mod(angles + np.pi, 2 * np.pi) - np.pi) < 1 * np.pi / 180)

        # less than 1 deg from pi
        is_anti_collinear = np.abs(np.abs(angles) - np.pi) < 1 * np.pi / 180
        factors[is_anti_collinear] *= -1
        is_multiple = np.abs(np.mod(factors + 0.5, 1) - 0.5) < 1e-2
        is_used = is_multiple & (is_collinear | is_anti_collinear)
        rounded_factors = np.round(factors[is_used]).reshape(-1, 1)
        ks2s_rescaled = 1 / rounded_factors * ks[is_used, :]
        ks_base[1, :] = np.mean(ks2s_rescaled, axis=0)
        break
    ks_base[ks_base[:, 0] < 0, :] *= -1
    return ks_base, auto_correlation_saved, center_peaks


def get_image_hexagonalicity(image, rotations=(60, 300)):
    image[image != 0] -= np.mean(image)
    std_image = np.std(image)
    if std_image == 0:
        return 0  # in case someting got realy wrong durring fit.
    image = image.astype(float) / std_image
    # plot_image(image, wait=True)
    hexagonalicity = 0
    for rot in rotations:
        rotated_image = scipy.ndimage.interpolation.rotate(
            image, rot, reshape=False)
        difference = image - rotated_image

        # max std of difference is 2 as std of image is 1
        relative_distance = np.std(difference) / 2
        # print("relative_distance", relative_distance)
        # plot_image(difference, wait=False, vmin=-1, vmax=1)
        hexagonalicity += (1 - relative_distance) / len(rotations)
    # print('hex=', hexagonalicity, "in", end - init)
    return hexagonalicity


def get_neg_hexagonalicity_with_mask(
    center,
    image,
    r1,
    r2,
    rotations=(60, 300)
):
    """return the negative of hexagonalicity of the image

    using an hexagonal mask defined by center, r1 and r2.

    Parameters
    ----------
    center : tuple(?)
        center of the hexagonal mask
    image : 2d-array(?)
        image used
    r1 : float(?)
        1st radius defining the hexagonal mask used
    r2 : float(?)
        2nd radius defining the hexagonal mask used (after r1 anticlockwise)
    rotations : iterable of floats(?)
        angles used in hexagonalicity computations

    Returns
    -------
    float
        negative hexagonalicity
    """
    image_center = (np.array(image.shape[::-1]) - 1) / 2
    displacement = image_center - center
#    plot_image(image)
    M = np.float32([[1, 0, displacement[0]], [0, 1, displacement[1]]])
    image = cv2.warpAffine(image, M, image.shape[::-1])
#    print("displacement=", displacement, "in", end - init)
#    plot_image(image)
    # vectors defining the hexagonal mask used
    r3 = -r1 + r2
    points = np.array((image_center + r1,
                       image_center + r2,
                       image_center + r3,
                       image_center - r1,
                       image_center - r2,
                       image_center - r3))
    pixels_x_min = int(np.floor(np.min(points[:, 0])))
    pixels_x_max = int(np.ceil(np.max(points[:, 0]) + 1))
    pixels_y_min = int(np.floor(np.min(points[:, 1])))
    pixels_y_max = int(np.ceil(np.max(points[:, 1]) + 1))
    if (pixels_x_max + pixels_x_min - 1) / 2 < image_center[0]:
        pixels_x_max = int(2 * image_center[0] - pixels_x_min + 1)
    elif (pixels_x_max + pixels_x_min - 1) / 2 > image_center[0]:
        pixels_x_min = int(2 * image_center[0] - pixels_x_max + 1)
    if (pixels_y_max + pixels_y_min - 1) / 2 < image_center[1]:
        pixels_y_max = int(2 * image_center[1] - pixels_y_min + 1)
    elif (pixels_y_max + pixels_y_min - 1) / 2 > image_center[1]:
        pixels_y_min = int(2 * image_center[1] - pixels_y_max + 1)
    image_crop = image[pixels_y_min:pixels_y_max, pixels_x_min:pixels_x_max]
#    plot_image(image_crop, wait=False)
    center_crop = np.array(image_center) - np.array(
        (pixels_x_min, pixels_y_min))
#    init = clock()
    mask_hexa = np.zeros_like(image_crop)
    mask_hexa = set_hexagon(mask_hexa, center=center_crop, r1=r1, r2=r2)
#    plot_image(image_crop * mask_hexa, wait=False)
#    print("mask in", end - init)
    return -get_image_hexagonalicity(
        image_crop * mask_hexa, rotations=rotations)


def reciprocal_to_lattice_space(a1, a2, shape):
    '''return solution to
        a1.dot(b1) = shape
        a2.dot(b1) = 0
        a1.dot(b2) = 0
        a2.dot(b2) = shape
    '''
    b1 = np.array((
        1 / (a1[0] - a1[1] / a2[1] * a2[0]),
        1 / (a1[1] - a1[0] / a2[0] * a2[1]))
    ) * shape
    b2 = np.array((
        1 / (a2[0] - a2[1] / a1[1] * a1[0]),
        1 / (a2[1] - a2[0] / a1[0] * a1[1]))
    ) * shape
    return (b1, b2)


def moments2D(inpData):
    """ Returns the (amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e)

    estimated from moments in the 2d input array Data

    from https://github.com/indiajoe/HandyTools4Astronomers
    """
    # Taking median of the 4 edges points as background
    bkg = np.median(np.hstack(
        (inpData[0, :], inpData[-1, :], inpData[:, 0], inpData[:, -1])))
    # Removing the background for calculating moments of pure 2D gaussian
    Data = np.ma.masked_less(inpData-bkg, 0)
    # We also masked any negative values before measuring moments
    amplitude = Data.max()
    total = float(Data.sum())
    Xcoords, Ycoords = np.indices(Data.shape)
    xcenter = (Xcoords*Data).sum()/total
    ycenter = (Ycoords*Data).sum()/total
    # Cut along the row of data near center of gaussian
    RowCut = Data[int(xcenter), :]
    # Cut along the column of data near center of gaussian
    ColumnCut = Data[:, int(ycenter)]
    xsigma = np.sqrt(np.ma.sum(
        ColumnCut * (np.arange(len(ColumnCut))-xcenter)**2)/ColumnCut.sum())
    ysigma = np.sqrt(np.ma.sum(
        RowCut * (np.arange(len(RowCut))-ycenter)**2)/RowCut.sum())
    # Ellipcity and position angle calculation
    Mxx = np.ma.sum((Xcoords-xcenter)*(Xcoords-xcenter) * Data) / total
    Myy = np.ma.sum((Ycoords-ycenter)*(Ycoords-ycenter) * Data) / total
    Mxy = np.ma.sum((Xcoords-xcenter)*(Ycoords-ycenter) * Data) / total
    e = np.sqrt((Mxx - Myy)**2 + (2*Mxy)**2) / (Mxx + Myy)
    pa = 0.5 * np.arctan(2*Mxy / (Mxx - Myy))
    rot = np.rad2deg(pa)
    return amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e


def gaussian_2d(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg):
    """ Returns a 2D Gaussian function with input parameters.
    rotation input rot should be in degress
    from https://github.com/indiajoe/HandyTools4Astronomers
    """
    # Now lets define the 2D gaussian function
    rot = np.deg2rad(rot)  # Converting to radians
    # Centers in rotated coordinates
    Xc = xcenter*np.cos(rot) - ycenter*np.sin(rot)
    Yc = xcenter*np.sin(rot) + ycenter*np.cos(rot)

    # Now lets define the 2D gaussian function
    def Gauss2D(x, y):
        """ Returns the values of the defined 2d gaussian at x,y """

        # X position in rotated coordinates
        xr = x * np.cos(rot) - y * np.sin(rot)
        yr = x * np.sin(rot) + y * np.cos(rot)
        return amplitude * np.exp(
            -(
                ((xr - Xc) / xsigma) ** 2 +
                ((yr - Yc) / ysigma) ** 2) / 2
            ) + bkg
    return Gauss2D


def fit_gauss_2d(data, initial_param=None):
    """ Fits 2D gaussian to data with optional Initial conditions

    initial_param=(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg)
    Example:
    >>> X,Y=np.indices((40,40),dtype=np.float)
    >>> data=np.exp(-(((X-25)/5)**2 +((Y-15)/10)**2)/2) + 1
    >>> FitGauss2D(data)
    (
        array([
            1.00000000e+00,
            2.50000000e+01,
            1.50000000e+01,
            5.00000000e+00,
            1.00000000e+01,
            2.09859373e-07,
            1]),
        2
    )

    from https://github.com/indiajoe/HandyTools4Astronomers
    """

    # Estimate the initial parameters from moments
    # and also set rot angle to be 0
    if initial_param is None:
        # Remove ellipticity from the end in parameter list
        initial_param = moments2D(data)[:-1]

    Xcoords, Ycoords = np.indices(data.shape)

    def errfun(ip):
        dXcoords = Xcoords - ip[1]
        dYcoords = Ycoords - ip[2]
        # Taking radius as the weights for least square fitting
        Weights = np.sqrt(np.square(dXcoords) + np.square(dYcoords) + 1e-6)
        # Taking a sqrt(weight) here so that
        # while scipy takes square of this array it will become 1/r weight.
        return np.ravel(
            (gaussian_2d(*ip)(*np.indices(data.shape)) - data) /
            np.sqrt(Weights)
        )
    p, success = scipy.optimize.leastsq(errfun, initial_param, maxfev=1000)
    return p, success
