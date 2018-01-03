import numbers
import numpy as np
import cv2  # opencv has a problem when imported after something with tensorflow, so I import it high up :-|
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
 from photutils import DAOStarFinder
from skimage.draw import polygon


def crop_image(image, crop_pixel1, crop_pixel2):
    """
    :param image: 2D array of the image data with a shape = (vertical size, horizontal size)
    :param crop_pixel1: iterable of 2 ints: position (pos_x , pos_y) of one corner of the returned region
    :param crop_pixel2: iterable of 2 ints: position (pos_x , pos_y) of the opposite corner of the returned region
    :return: the cropped image, the 1st corner and the opposite corner positions
    """
    if type(image) is str:
        image = fits.open(image)[0].data
    if type(image) is not np.ndarray:
        raise AttributeError("image must be a filename or a numpy array.")
    if crop_pixel1 is not list and len(crop_pixel1) is not 2:
        raise AttributeError("crop_pixel1 must be a list of 2 ints", "(position in px of one corner of the crop area).")
    if crop_pixel2 is not list and len(crop_pixel1) is not 2:
        raise AttributeError("crop_pixel2 must be a list of 2 ints",
                             "(position in px of the other corner of the crop area).")
    crop_xmin, crop_ymin = crop_pixel1
    crop_xmax, crop_ymax = crop_pixel2
    if not isinstance(crop_xmin, numbers.Integral) or not isinstance(crop_ymin, numbers.Integral):
        print('Warning in crop_image(): crop_pixel1 does not contain ints')
        crop_xmin = int(crop_xmin)
        crop_ymin = int(crop_ymin)
    if not isinstance(crop_xmax, numbers.Integral) or not isinstance(crop_ymax, numbers.Integral):
        print('Warning in crop_image(): crop_pixel2 does not contain ints')
        crop_xmax = int(crop_xmax)
        crop_ymax = int(crop_ymax)
    npixel_y, npixel_x = image.shape
    if not (0 <= crop_xmin < npixel_x and 0 <= crop_ymin < npixel_y):
        raise AttributeError("invalid crop_pixel position:", crop_pixel1, crop_pixel2, "image size is: ", image.shape)
    if not (0 < crop_xmax < npixel_x and 0 < crop_ymax < npixel_y):
        raise AttributeError("invalid crop_pixel position:", crop_pixel1, crop_pixel2, "image size is: ", image.shape)
    if crop_xmin > crop_xmax:
        crop_xmin, crop_xmax = crop_xmax, crop_xmin
    if crop_ymin > crop_ymax:
        crop_ymin, crop_ymax = crop_ymax, crop_ymin
    crop_pixel1 = crop_xmin, crop_ymin
    crop_pixel2 = crop_xmax, crop_ymax
    return image[crop_ymin:crop_ymax, crop_xmin:crop_xmax], crop_pixel1, crop_pixel2


def average_images(images):
    if type(images) is not list:
        print('ERROR in average_images(): images must be a list of filenames or numpy.ndarray')
        return None
    nimage = len(images)
    if nimage == 0:
        print('ERROR in average_images(): empty list of images')
        return None
    if type(images[0]) is not np.ndarray:
        print('ERROR in average_images(): images must be numpy.ndarray')
        return None
    first_image_shape = images[0].shape
    image_average = np.zeros(first_image_shape)
    for image in images:
        if type(image) is not np.ndarray:
            print('WARNING in average_images(): image is not a numpy.ndarray')
            continue
        if first_image_shape != image.shape:
            print('ERROR in average_images(): images are not of same sizes')
            return None
        image_average += image / nimage
    return image_average


def set_circle(image, center=(0, 0), radius=10, value=1):
    for ix in range(int(center[0] - radius - 1), int(center[0] + radius + 1)):
        for iy in range(int(center[1] - radius - 1), int(center[1] + radius + 1)):
            if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]):
                continue  # out of image border
            if np.sqrt((ix - center[0]) * (ix - center[0]) + (iy - center[1]) * (iy - center[1])) < radius:
                image[iy, ix] = value
    return image


def make_repetitive_mask(shape, radius, v1, v2, center, nrepetition=100):
    mask = np.zeros(shape)
    for nv1 in range(-int(nrepetition/2), int(nrepetition/2)):
        for nv2 in range(-int(nrepetition/2), int(nrepetition/2)):
            center_circle = center + nv1 * v1 + nv2 * v2
            mask = set_circle(mask, center=center_circle.reshape((2,)), radius=radius, value=1)
    return mask


def set_parallelogram(image, center=(0, 0), k1=(1, 0), k2=(0, 1), value=1):
    origin = center - k1 / 2 - k2 / 2
    r = np.array((origin[0], origin[0]+k1[0], origin[0]+k1[0]+k2[0], origin[0]+k2[0]))
    c = np.array((origin[1], origin[1]+k1[1], origin[1]+k1[1]+k2[1], origin[1]+k2[1]))
    rr, cc = polygon(r, c, (image.shape[1], image.shape[0]))
    image[cc, rr] = value
    return image


def get_consecutive_hex_radius(r1, r2):  # set r2 to be the next radius after r1 anticlockwise
    r1 = np.array(r1)
    r2 = np.array(r2)
    # set r2 to be the next radius after r1 (anticlockwise)
    angle_r1 = np.angle(r1[0] + 1j * r1[1])
    angle_r2 = np.angle(r2[0] + 1j * r2[1])
    angle_diff = np.mod(angle_r2 - angle_r1, 2 * np.pi)
    resol = 1e-2
    if np.abs(angle_diff) < resol or np.abs(angle_diff - np.pi) < resol or np.abs(angle_diff) > 2 * np.pi - resol:
        raise AttributeError('r1 and r2 are collinear')
    if np.pi / 2 < angle_diff < np.pi:
        r2 = r1 + r2
    elif np.pi < angle_diff < 3 / 2 * np.pi:
        r2 = -r2
    elif angle_diff > 3 / 2 * np.pi:
        r2 = r1 - r2
    return r1, r2


def set_hexagon(image, center=(0, 0), r1=(1, 0), r2=(0.5, np.sqrt(3) / 2), value=1):
    r1, r2 = get_consecutive_hex_radius(r1, r2)
    # set of hexagon's vertexes
    points = np.array((center + r1 ,
                       center + r2,
                       center + r2 - r1,
                       center - r1,
                       center - r2,
                       center - r2 + r1))
    rr, cc = polygon(points[:, 0], points[:, 1], (image.shape[1], image.shape[0]))
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


def get_peaks_separation(fft_image_shifted, center=None, crop_range=None, radius_removed=20):
    if center is None:
        center = (np.array(fft_image_shifted.shape[::-1]) - 1) / 2
    fft_image_shifted = set_circle(fft_image_shifted, center=center, radius=radius_removed, value=0)
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
    auto_correlation = signal.fftconvolve(crop_fft, crop_fft[::-1, ::-1], mode='same')
    auto_correlation_saved = auto_correlation
    """
    auto_correlation_saved = copy.deepcopy(auto_correlation)
    center_peaks = []
    for i in range(81):
        auto_correlation[auto_correlation < 0] = 0
        max_pos = np.argmax(auto_correlation)
        [max_pos_y, max_pos_x] = np.unravel_index(max_pos, auto_correlation.shape)
        region_fit = auto_correlation[max_pos_y - 20:max_pos_y + 20, max_pos_x - 20:max_pos_x + 20]
        init_amplitude = auto_correlation[max_pos_y, max_pos_x] - np.mean(region_fit)
        init_param = (init_amplitude, 20, 20, 1, 1, 0, np.mean(region_fit))
        fit_result, success = FitGauss2D(region_fit.transpose(), ip=init_param)
        amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg = fit_result
        if success != 1 or xsigma > 5 or ysigma > 5:
            auto_correlation = set_circle(auto_correlation, center=(max_pos_x, max_pos_y), radius=10, value=0)
            # print('WARNING: fit of the peak failed in get_peaks_separation()')
            continue
        gaussian_function = Gaussian2D(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg)
        center_peaks.append(np.array((xcenter + max_pos_x - 20, ycenter + max_pos_y - 20)))
        for ix, x in enumerate(range(max_pos_x - 20, max_pos_x + 20)):
            for iy, y in enumerate(range(max_pos_y - 20, max_pos_y + 20)):
                auto_correlation[y, x] -= gaussian_function(ix, iy)
    center_peaks = np.array(center_peaks)
    """
    auto_correlation[auto_correlation < 0] = 0
    mean, median, std = sigma_clipped_stats(auto_correlation, sigma=3.0, iters=5)
    baseline_sub = auto_correlation - median
    baseline_sub[baseline_sub < 0] = 0
    daofind = DAOStarFinder(fwhm=3.0, threshold=std)
    sources = daofind(baseline_sub)
    center_peaks = np.array([sources['xcentroid'], sources['ycentroid']]).transpose()
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
    is_collinear = (np.abs(np.mod(angles+np.pi, 2*np.pi)-np.pi) < 1*np.pi/180)  # less than 1 deg from 0
    is_anti_collinear = np.abs(np.abs(angles)-np.pi) < 1*np.pi/180  # less than 1 deg from pi
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
        is_collinear = (np.abs(np.mod(angles + np.pi, 2 * np.pi) - np.pi) < 1 * np.pi / 180)  # less than 1 deg from 0
        is_anti_collinear = np.abs(np.abs(angles) - np.pi) < 1 * np.pi / 180  # less than 1 deg from pi
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
    # init = clock()
    image[image != 0] -= np.mean(image)
    std_image = np.std(image)
    if std_image == 0:
        return 0  #in case someting got realy wrong durring fit.
    image = image.astype(float) / std_image
    # plot_image(image, wait=True)
    hexagonalicity = 0
    for rot in rotations:
        rotated_image = scipy.ndimage.interpolation.rotate(image, rot, reshape=False)
        difference = image - rotated_image
        relative_distance = np.std(difference) / 2  # max std of difference is 2 as std of image is 1
        # print("relative_distance", relative_distance)
        # plot_image(difference, wait=False, vmin=-1, vmax=1)
        hexagonalicity += (1 - relative_distance) / len(rotations)
    # end = clock()
    # print('hex=', hexagonalicity, "in", end - init)
    return hexagonalicity


def get_neg_hexagonalicity_with_mask(center, image, r1, r2, rotations=(60, 300)):
    """
    return the negative of hexagonalicity of the image using an hexagonal mask defined by center, r1 and r2.
    :param center: center of the hexagonal mask
    :param image: image used
    :param r1: 1st radius defining the hexagonal mask used
    :param r2: 2nd radius defining the hexagonal mask used (after r1 anticlockwise)
    :param rotations: angles used in hexagonalicity computations
    :return: -hexagonalicity
    """
    image_center = (np.array(image.shape[::-1]) - 1) / 2
    displacement = image_center - center
#    plot_image(image)
#    init = clock()
    M = np.float32([[1, 0, displacement[0]], [0, 1, displacement[1]]])
    image = cv2.warpAffine(image, M, image.shape[::-1])
#    end = clock()
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
    center_crop = np.array(image_center) - np.array((pixels_x_min, pixels_y_min))
#    init = clock()
    mask_hexa = np.zeros_like(image_crop)
    mask_hexa = set_hexagon(mask_hexa, center=center_crop, r1=r1, r2=r2)
#    plot_image(image_crop * mask_hexa, wait=False)
#    end = clock()
#    print("mask in", end - init)
    return -get_image_hexagonalicity(image_crop * mask_hexa, rotations=rotations)


def reciprocal_to_lattice_space(a1, a2, shape):
    # solution to a1.dot(b1) = shape, a2.dot(b1) = 0, a1.dot(b2) = 0, a2.dot(b2) = shape,
    b1 = np.array((1 / (a1[0] - a1[1] / a2[1] * a2[0]), 1 / (a1[1] - a1[0] / a2[0] * a2[1]))) * shape
    b2 = np.array((1 / (a2[0] - a2[1] / a1[1] * a1[0]), 1 / (a2[1] - a2[0] / a1[0] * a1[1]))) * shape
    return (b1, b2)


def moments2D(inpData):  # from https://github.com/indiajoe/HandyTools4Astronomers
    """ Returns the (amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e) estimated from moments in the 2d input array Data """
    # Taking median of the 4 edges points as background
    bkg = np.median(np.hstack((inpData[0, :], inpData[-1, :], inpData[:, 0], inpData[:, -1])))
    Data=  np.ma.masked_less(inpData-bkg, 0)   #Removing the background for calculating moments of pure 2D gaussian
    # We also masked any negative values before measuring moments
    amplitude = Data.max()
    total = float(Data.sum())
    Xcoords, Ycoords = np.indices(Data.shape)
    xcenter = (Xcoords*Data).sum()/total
    ycenter = (Ycoords*Data).sum()/total
    RowCut = Data[int(xcenter), :]  # Cut along the row of data near center of gaussian
    ColumnCut = Data[:, int(ycenter)]  # Cut along the column of data near center of gaussian
    xsigma = np.sqrt(np.ma.sum(ColumnCut * (np.arange(len(ColumnCut))-xcenter)**2)/ColumnCut.sum())
    ysigma = np.sqrt(np.ma.sum(RowCut * (np.arange(len(RowCut))-ycenter)**2)/RowCut.sum())
    #Ellipcity and position angle calculation
    Mxx = np.ma.sum((Xcoords-xcenter)*(Xcoords-xcenter) * Data) / total
    Myy = np.ma.sum((Ycoords-ycenter)*(Ycoords-ycenter) * Data) / total
    Mxy = np.ma.sum((Xcoords-xcenter)*(Ycoords-ycenter) * Data) / total
    e = np.sqrt((Mxx - Myy)**2 + (2*Mxy)**2) / (Mxx + Myy)
    pa = 0.5 * np.arctan(2*Mxy / (Mxx - Myy))
    rot = np.rad2deg(pa)
    return amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e


def Gaussian2D(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg):  # from https://github.com/indiajoe/HandyTools4Astronomers
    """ Returns a 2D Gaussian function with input parameters. rotation input rot should be in degress """
    rot=np.deg2rad(rot)  # Converting to radians
    Xc=xcenter*np.cos(rot) - ycenter*np.sin(rot)  # Centers in rotated coordinates
    Yc=xcenter*np.sin(rot) + ycenter*np.cos(rot)
    # Now lets define the 2D gaussian function

    def Gauss2D(x, y):
        """ Returns the values of the defined 2d gaussian at x,y """
        xr = x * np.cos(rot) - y * np.sin(rot)  # X position in rotated coordinates
        yr = x * np.sin(rot) + y * np.cos(rot)
        return amplitude * np.exp(-(((xr - Xc) / xsigma) ** 2 +((yr - Yc) / ysigma) ** 2) / 2) + bkg
    return Gauss2D


def FitGauss2D(Data, ip=None):  # from https://github.com/indiajoe/HandyTools4Astronomers
    """ Fits 2D gaussian to Data with optional Initial conditions ip=(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg)
    Example:
    >>> X,Y=np.indices((40,40),dtype=np.float)
    >>> Data=np.exp(-(((X-25)/5)**2 +((Y-15)/10)**2)/2) + 1
    >>> FitGauss2D(Data)
    (array([  1.00000000e+00,   2.50000000e+01,   1.50000000e+01, 5.00000000e+00,   1.00000000e+01,   2.09859373e-07, 1]), 2)
     """
    if ip is None:   # Estimate the initial parameters form moments and also set rot angle to be 0
        ip=moments2D(Data)[:-1]   # Remove ellipticity from the end in parameter list
    Xcoords, Ycoords = np.indices(Data.shape)
    def errfun(ip):
        dXcoords = Xcoords - ip[1]
        dYcoords = Ycoords - ip[2]
        # Taking radius as the weights for least square fitting
        Weights = np.sqrt(np.square(dXcoords) + np.square(dYcoords) + 1e-6)
        # Taking a sqrt(weight) here so that while scipy takes square of this array it will become 1/r weight.
        return np.ravel((Gaussian2D(*ip)(*np.indices(Data.shape)) - Data)/np.sqrt(Weights))
    p, success = scipy.optimize.leastsq(errfun, ip, maxfev=1000)
    return p, success
