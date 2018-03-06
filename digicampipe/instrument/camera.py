# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.utils import lazyproperty
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import splrep, splev
import scipy.ndimage as ndimage
from ctapipe.utils.linalg import rotation_matrix_2d

__all__ = ['CameraGeometry', 'CameraCalibration']

logger = logging.getLogger(__name__)

# dictionary to convert number of pixels to camera + the focal length of the
# telescope into a camera type for use in `CameraGeometry.guess()`
#     Key = (num_pix, focal_length_in_meters)
#     Value = (type, subtype, pixtype, pixrotation, camrotation)
_CAMERA_GEOMETRY_TABLE = {
    (2048, 2.3):
        ('SST', 'CHEC', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 2.2):
        ('SST', 'CHEC', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 36.0):
        ('LST', 'HESS-II', 'hexagonal', 0 * u.degree, 0 * u.degree),
    (960, None):
        ('MST', 'HESS-I', 'hexagonal', 0 * u.degree, 0 * u.degree),
    (1855, 16.0):
        ('MST', 'NectarCam', 'hexagonal', 0 * u.degree, -100.893 * u.degree),
    (1855, 28.0):
        ('LST', 'LSTCam', 'hexagonal', 0. * u.degree, -100.893 * u.degree),
    (1296, None):
        ('SST', 'DigiCam', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (1764, None):
        ('MST', 'FlashCam', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (2368, None):
        ('SST', 'ASTRICam', 'rectangular', 0 * u.degree, 0 * u.degree),
    (11328, None):
        ('SCT', 'SCTCam', 'rectangular', 0 * u.degree, 0 * u.degree),
}


class CameraCalibration:
    def __init__(
        self,
        gain,
        sigma_e,
        xt,
        charge_reconstruction_options,
        cell_capacitance=85. * 1E-15,
        bias_resistance=10. * 1E3
    ):

        self.gain = gain
        self.sigma_e = sigma_e
        self.xt = xt
        self.cell_capacitance = cell_capacitance
        self.bias_resistance = bias_resistance
        self.charge_reconstruction_options = charge_reconstruction_options
        self.lut_std = np.array([
            1.10364884, 1.12430139, 1.2435541, 1.36286533, 1.59703604,
            1.80362736, 2.07318656, 2.50829027, 2.92629147, 3.44292195,
            4.0800227, 4.79139778, 5.63669767, 6.53108791, 7.64749998,
            8.65923108, 9.57379588, 10.37193237, 10.88006117, 11.19791948,
            11.10610816, 10.95768853, 10.57739894, 10.37384412, 10.18064523,
            10.75009621, 11.57963516, 13.24342791, 15.57271873, 18.24045984])
        self.lut_nsb_rate = np.array([
            1.00000000e-03, 1.45222346e-03, 2.10895298e-03, 3.06267099e-03,
            4.44768267e-03, 6.45902911e-03, 9.37995361e-03, 1.36217887e-02,
            1.97818811e-02, 2.87277118e-02, 4.17190571e-02, 6.05853935e-02,
            8.79835297e-02, 1.27771746e-01, 1.85553127e-01, 2.69464604e-01,
            3.91322820e-01, 5.68288180e-01, 8.25281427e-01, 1.19849305e+00,
            1.74047972e+00, 2.52756549e+00, 3.67058990e+00, 5.33051677e+00,
            7.74110151e+00, 1.12418092e+01, 1.63256191e+01, 2.37084470e+01,
            3.44299630e+01, 5.00000000e+01])
        self.lut_baseline_shift = np.array([
            500.10844, 500.13456, 500.19772, 500.2697, 500.44418, 500.6094,
            500.87226, 501.33124, 501.8768, 502.71168, 503.9574, 505.51968,
            507.92038, 511.24914, 515.572, 521.41952, 528.5894, 537.03064,
            547.04614, 557.76222, 568.03408, 577.88066, 586.55614, 593.50884,
            598.97944, 603.33174, 606.54438, 608.84342, 610.4056, 611.94482])

        self.lut_baseline_shift -= 500.0

        self.spline_gain_drop = splrep(
            self.lut_baseline_shiftbaseline_shift,
            self.gain_drop(self.lut_nsb_rate)
        )
        self.spline_nsb_rate = splrep(
            self.lut_baseline_shiftbaseline_shift,
            self.lut_nsb_rate
        )

    def extract_charge(
        self,
        data,
        timing_mask,
        timing_mask_edge,
        peak,
        window_start,
        threshold_saturation
    ):
        """
        Extract the charge.
           - check which pixels are saturated
           - get the local maximum within the timing mask and check
             if it is not at the edge of the mask
           - move window_start from the maximum
        :param data:
        :param timing_mask:
        :param timing_mask_edge:
        :param peak_position:
        :param options:
        :param integration_type:
        :return:
        """

        is_saturated = np.max(data, axis=-1) > threshold_saturation
        local_max = np.argmax(np.multiply(data, timing_mask), axis=1)
        local_max_edge = np.argmax(np.multiply(data, timing_mask_edge), axis=1)
        ind_max_at_edge = (local_max == local_max_edge)
        local_max[ind_max_at_edge] = peak[ind_max_at_edge] - window_start
        index_max = (np.arange(0, data.shape[0]), local_max,)
        ind_with_lt_th = data[index_max] < 10.
        local_max[ind_with_lt_th] = peak[ind_with_lt_th] - window_start
        local_max[local_max < 0] = 0
        index_max = (np.arange(0, data.shape[0]), local_max,)
        charge = data[index_max]
        if np.any(is_saturated):  ## TODO, find a better evaluation that it is saturated
            sat_indices = tuple(np.where(is_saturated)[0])
            _data = data[sat_indices, ...]
            charge[sat_indices, ...] = np.apply_along_axis(contiguous_regions, 1, _data)

        return charge, index_max

    def contiguous_regions(data):
        """Finds contiguous True regions of the boolean array "condition". Returns
        a 2D array where the first column is the start index of the region and the
        second column is the end index."""
        condition = data > 0
        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero()

        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size]  # Edit

        # Reshape the result into two columns
        idx.shape = (-1, 2)
        val = 0.
        for start, stop in idx:
            sum_tmp = np.sum(data[start:stop])
            if val < sum_tmp: val = sum_tmp
        return val

    def fake_timing_hist(n_samples, timing_width, central_sample):
        """
        Create a timing array based on options.central_sample and options.timing_width
        :param options:
        :param n_samples:
        :return:
        """
        timing = np.zeros((1296, n_samples + 1,), dtype=float)
        timing[..., int(central_sample - timing_width):int(central_sample + timing_width)] = 1.
        return timing

    def generate_timing_mask(window_start, window_width, peak_positions):
        """
        Generate mask arround the possible peak position
        :param peak_positions:
        :return:
        """
        peak = np.argmax(peak_positions, axis=1)
        mask = (peak_positions.T / np.sum(peak_positions, axis=1)).T > 1e-3
        mask_window = mask + np.append(mask[..., 1:], np.zeros((peak_positions.shape[0], 1), dtype=bool), axis=1) + \
                      np.append(np.zeros((peak_positions.shape[0], 1), dtype=bool), mask[..., :-1], axis=1)
        mask_windows_edge = mask_window * ~mask
        mask_window = mask_window[..., :-1]
        mask_windows_edge = mask_windows_edge[..., :-1]
        shift = window_start  # window_width - int(np.floor(window_width/2))+window_start
        missing = mask_window.shape[1] - (window_width - 1)
        mask_window = mask_window[..., shift:]
        missing = mask_window.shape[1] - missing
        mask_window = mask_window[..., :-missing]
        mask_windows_edge = mask_windows_edge[..., shift:]
        mask_windows_edge = mask_windows_edge[..., :-missing]
        return peak, mask_window, mask_windows_edge


class CameraGeometry:
    """`CameraGeometry` is a class that stores information about a
    Cherenkov Camera that us useful for imaging algorithms and
    displays. It contains lists of pixel positions, areas, pixel
    shapes, as well as a neighbor (adjacency) list and matrix for each pixel.
    In general the neighbor_matrix attribute should be used in any algorithm
    needing pixel neighbors, since it is much faster. See for example
    `ctapipe.image.tailcuts_clean`

    The class is intended to be generic, and work with any Cherenkov
    Camera geometry, including those that have square vs hexagonal
    pixels, gaps between pixels, etc.

    You can construct a CameraGeometry either by specifying all data,
    or using the `CameraGeometry.guess()` constructor, which takes metadata
    like the pixel positions and telescope focal length to look up the rest
    of the data. Note that this function is memoized, so calling it multiple
    times with the same inputs will give back the same object (for speed).

    Parameters
    ----------
    self: type
        description
    cam_id: camera id name or number
        camera identification string
    pix_id: array(int)
        pixels id numbers
    pix_x: array with units
        position of each pixel (x-coordinate)
    pix_y: array with units
        position of each pixel (y-coordinate)
    pix_area: array(float)
        surface area of each pixel, if None will be calculated
    neighbors: list(arrays)
        adjacency list for each pixel
    pix_type: string
        either 'rectangular' or 'hexagonal'
    pix_rotation: value convertable to an `astropy.coordinates.Angle`
        rotation angle with unit (e.g. 12 * u.deg), or "12d"
    cam_rotation: overall camera rotation with units
    """

    _geometry_cache = {}  # dictionary CameraGeometry instances for speed

    def __init__(self, cam_id, pix_id, pix_x, pix_y, pix_area, pix_type,
                 pix_rotation="0d", cam_rotation="0d",
                 neighbors=None, apply_derotation=True):

        self.cam_id = cam_id
        self.pix_id = pix_id
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.pix_area = pix_area
        self.pix_type = pix_type
        self.pix_rotation = Angle(pix_rotation)
        self.cam_rotation = Angle(cam_rotation)
        self._precalculated_neighbors = neighbors

        if self.pix_area is None:
            self.pix_area = CameraGeometry._calc_pixel_area(pix_x, pix_y,
                                                            pix_type)

        if apply_derotation:
            # todo: this should probably not be done, but need to fix
            # GeometryConverter and reco algorithms if we change it.
            if len(pix_x.shape) == 1:
                self.rotate(cam_rotation)

    def __eq__(self, other):
        return ((self.cam_id == other.cam_id)
                and (self.pix_x == other.pix_x).all()
                and (self.pix_y == other.pix_y).all()
                and (self.pix_type == other.pix_type)
                and (self.pix_rotation == other.pix_rotation)
                and (self.pix_type == other.pix_type)
                )

    @classmethod
    @u.quantity_input
    def guess(cls, pix_x: u.m, pix_y: u.m, optical_foclen: u.m,
              apply_derotation=True):
        """
        Construct a `CameraGeometry` by guessing the appropriate quantities
        from a list of pixel positions and the focal length.
        """
        # only construct a new one if it has never been constructed before,
        # to speed up access. Otherwise return the already constructed instance
        # the identifier uses the values of pix_x (which are converted to a
        # string to make them hashable) and the optical_foclen. So far,
        # that is enough to uniquely identify a geometry.
        identifier = (pix_x.value.tostring(), optical_foclen)
        if identifier in CameraGeometry._geometry_cache:
            return CameraGeometry._geometry_cache[identifier]

        # now try to determine the camera type using the map defined at the
        # top of this file.

        tel_type, cam_id, pix_type, pix_rotation, cam_rotation = \
            _guess_camera_type(len(pix_x), optical_foclen)

        area = cls._calc_pixel_area(pix_x, pix_y, pix_type)

        instance = cls(
            cam_id=cam_id,
            pix_id=np.arange(len(pix_x)),
            pix_x=pix_x,
            pix_y=pix_y,
            pix_area=np.ones(pix_x.shape) * area,
            neighbors=None,
            pix_type=pix_type,
            pix_rotation=Angle(pix_rotation),
            cam_rotation=Angle(cam_rotation),
            apply_derotation=apply_derotation
        )

        CameraGeometry._geometry_cache[identifier] = instance
        return instance

    @staticmethod
    def _calc_pixel_area(pix_x, pix_y, pix_type):
        """ recalculate pixel area based on the pixel type and layout

        Note this will not work on cameras with varying pixel sizes.
        """

        dist = _get_min_pixel_seperation(pix_x, pix_y)

        if pix_type.startswith('hex'):
            rad = dist / np.sqrt(3)  # radius to vertex of hexagon
            area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
        elif pix_type.startswith('rect'):
            area = dist ** 2
        else:
            raise KeyError("unsupported pixel type")

        return np.ones(pix_x.shape) * area

    def to_table(self):
        """ convert this to an `astropy.table.Table` """
        # currently the neighbor list is not supported, since
        # var-length arrays are not supported by astropy.table.Table
        return Table([self.pix_id, self.pix_x, self.pix_y, self.pix_area],
                     names=['pix_id', 'pix_x', 'pix_y', 'pix_area'],
                     meta=dict(PIX_TYPE=self.pix_type,
                               TAB_TYPE='ctapipe.instrument.CameraGeometry',
                               TAB_VER='1.0',
                               CAM_ID=self.cam_id,
                               PIX_ROT=self.pix_rotation.deg,
                               CAM_ROT=self.cam_rotation.deg,
                               ))

    @classmethod
    def from_table(cls, url_or_table, **kwargs):
        """
        Load a CameraGeometry from an `astropy.table.Table` instance or a
        file that is readable by `astropy.table.Table.read()`

        Parameters
        ----------
        url_or_table: string or astropy.table.Table
            either input filename/url or a Table instance

        format: str
            astropy.table format string (e.g. 'ascii.ecsv') in case the
            format cannot be determined from the file extension

        kwargs: extra keyword arguments
            extra arguments passed to `astropy.table.read()`, depending on
            file type (e.g. format, hdu, path)


        """

        tab = url_or_table
        if not isinstance(url_or_table, Table):
            tab = Table.read(url_or_table, **kwargs)

        return cls(
            cam_id=tab.meta.get('CAM_ID', 'Unknown'),
            pix_id=tab['pix_id'],
            pix_x=tab['pix_x'].quantity,
            pix_y=tab['pix_y'].quantity,
            pix_area=tab['pix_area'].quantity,
            pix_type=tab.meta['PIX_TYPE'],
            pix_rotation=Angle(tab.meta['PIX_ROT'] * u.deg),
            cam_rotation=Angle(tab.meta['CAM_ROT'] * u.deg),
        )

    def __repr__(self):
        return "CameraGeometry(cam_id='{cam_id}', pix_type='{pix_type}', " \
               "npix={npix}, cam_rot={camrot}, pix_rot={pixrot})".format(
            cam_id=self.cam_id,
            pix_type=self.pix_type,
            npix=len(self.pix_id),
            pixrot=self.pix_rotation,
            camrot=self.cam_rotation
        )

    def __str__(self):
        return self.cam_id

    @lazyproperty
    def neighbors(self):
        """" only calculate neighbors when needed or if not already
        calculated"""

        # return pre-calculated ones (e.g. those that were passed in during
        # the object construction) if they exist
        if self._precalculated_neighbors is not None:
            return self._precalculated_neighbors

        # otherwise compute the neighbors from the pixel list
        dist = _get_min_pixel_seperation(self.pix_x, self.pix_y)

        neighbors = _find_neighbor_pixels(
            self.pix_x.value,
            self.pix_y.value,
            rad=1.4 * dist.value
        )

        return neighbors

    @lazyproperty
    def neighbor_matrix(self):
        return _neighbor_list_to_matrix(self.neighbors)

    @lazyproperty
    def neighbor_matrix_where(self):
        """
        Obtain a 2D array, where each row is [pixel index, one neighbour
        of that pixel].

        Returns
        -------
        ndarray
        """
        return np.ascontiguousarray(np.array(np.where(self.neighbor_matrix)).T)

    def rotate(self, angle):
        """rotate the camera coordinates about the center of the camera by
        specified angle. Modifies the CameraGeometry in-place (so
        after this is called, the pix_x and pix_y arrays are
        rotated.

        Notes
        -----

        This is intended only to correct simulated data that are
        rotated by a fixed angle.  For the more general case of
        correction for camera pointing errors (rotations,
        translations, skews, etc), you should use a true coordinate
        transformation defined in `ctapipe.coordinates`.

        Parameters
        ----------

        angle: value convertable to an `astropy.coordinates.Angle`
            rotation angle with unit (e.g. 12 * u.deg), or "12d"

        """
        rotmat = rotation_matrix_2d(angle)
        rotated = np.dot(rotmat.T, [self.pix_x.value, self.pix_y.value])
        self.pix_x = rotated[0] * self.pix_x.unit
        self.pix_y = rotated[1] * self.pix_x.unit
        self.pix_rotation -= Angle(angle)
        self.cam_rotation -= Angle(angle)

    @classmethod
    def make_rectangular(cls, npix_x=40, npix_y=40, range_x=(-0.5, 0.5),
                         range_y=(-0.5, 0.5)):
        """Generate a simple camera with 2D rectangular geometry.

        Used for testing.

        Parameters
        ----------
        npix_x : int
            number of pixels in X-dimension
        npix_y : int
            number of pixels in Y-dimension
        range_x : (float,float)
            min and max of x pixel coordinates in meters
        range_y : (float,float)
            min and max of y pixel coordinates in meters

        Returns
        -------
        CameraGeometry object

        """
        bx = np.linspace(range_x[0], range_x[1], npix_x)
        by = np.linspace(range_y[0], range_y[1], npix_y)
        xx, yy = np.meshgrid(bx, by)
        xx = xx.ravel() * u.m
        yy = yy.ravel() * u.m

        ids = np.arange(npix_x * npix_y)
        rr = np.ones_like(xx).value * (xx[1] - xx[0]) / 2.0

        return cls(cam_id=-1,
                   pix_id=ids,
                   pix_x=xx * u.m,
                   pix_y=yy * u.m,
                   pix_area=(2 * rr) ** 2,
                   neighbors=None,
                   pix_type='rectangular')


# ======================================================================
# utility functions:
# ======================================================================

def _get_min_pixel_seperation(pix_x, pix_y):
    """
    Obtain the minimum seperation between two pixels on the camera

    Parameters
    ----------
    pix_x : array_like
        x position of each pixel
    pix_y : array_like
        y position of each pixels

    Returns
    -------
    pixsep : astropy.units.Unit

    """
    #    dx = pix_x[1] - pix_x[0]    <=== Not adjacent for DC-SSTs!!
    #    dy = pix_y[1] - pix_y[0]

    dx = pix_x - pix_x[0]
    dy = pix_y - pix_y[0]
    pixsep = np.min(np.sqrt(dx ** 2 + dy ** 2)[1:])
    return pixsep


def _find_neighbor_pixels(pix_x, pix_y, rad):
    """use a KD-Tree to quickly find nearest neighbors of the pixels in a
    camera. This function can be used to find the neighbor pixels if
    they are not already present in a camera geometry file.

    Parameters
    ----------
    pix_x : array_like
        x position of each pixel
    pix_y : array_like
        y position of each pixels
    rad : float
        radius to consider neighbor it should be slightly larger
        than the pixel diameter.

    Returns
    -------
    array of neighbor indices in a list for each pixel

    """

    points = np.array([pix_x, pix_y]).T
    indices = np.arange(len(pix_x))
    kdtree = KDTree(points)
    neighbors = [kdtree.query_ball_point(p, r=rad) for p in points]
    for nn, ii in zip(neighbors, indices):
        nn.remove(ii)  # get rid of the pixel itself
    return neighbors


def _guess_camera_type(npix, optical_foclen):
    global _CAMERA_GEOMETRY_TABLE

    try:
        return _CAMERA_GEOMETRY_TABLE[(npix, None)]
    except KeyError:
        return _CAMERA_GEOMETRY_TABLE.get((npix, round(optical_foclen.value, 1)),
                                          ('unknown', 'unknown', 'hexagonal',
                                           0 * u.degree, 0 * u.degree))


def _neighbor_list_to_matrix(neighbors):
    """
    convert a neighbor adjacency list (list of list of neighbors) to a 2D
    numpy array, which is much faster (and can simply be multiplied)
    """

    npix = len(neighbors)
    neigh2d = np.zeros(shape=(npix, npix), dtype=np.bool)

    for ipix, neighbors in enumerate(neighbors):
        for jn, neighbor in enumerate(neighbors):
            neigh2d[ipix, neighbor] = True

    return neigh2d
