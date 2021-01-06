import numpy as np
from ctapipe.core.container import Container, Field
import astropy.units as u

import numpy as np
from ctapipe.coordinates import CameraFrame
import astropy.units as u
from astropy.utils import deprecated
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import Time
from . import disp
from warnings import warn

location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg,
                                       2184 * u.m)  # position of the LST1
obstime = Time('2018-11-01T02:00')
horizon_frame = AltAz(location=location, obstime=obstime)

def alt_to_theta(alt):
    """Transforms altitude (angle from the horizon upwards) to theta
    (angle from z-axis) for simtel array coordinate systems
    Parameters:
    -----------
    alt: float
    Returns:
    --------
    float: theta

    """

    return (90 * u.deg - alt).to(alt.unit)


def az_to_phi(az):
    """Transforms azimuth (angle from north towards east)
    to phi (angle from x-axis towards y-axis)
    for simtel array coordinate systems
    Parameters:
    -----------
    az: float

    Returns:
    --------
    az: float
    """
    return -az


@deprecated("09/07/2019",
            message="This is a custom implementation. Use `sky_to_camera` that relies on astropy")
def cal_cam_source_pos(mc_alt, mc_az, mc_alt_tel, mc_az_tel, focal_length):
    """Transform Alt-Az source position into Camera(x,y) coordinates
    source position.

    Parameters:
    -----------
    mc_alt: float
    Alt coordinate of the event
    mc_az: float
    Az coordinate of the event
    mc_alt_tel: float
    Alt coordinate of the telescope pointing
    mc_az_tel: float
    Az coordinate of the telescope pointing
    focal_length: float
    Focal length of the telescope

    Returns:
    --------
    float: source_x1,
    float: source_x2
    """

    mc_alt = alt_to_theta(mc_alt * u.rad).value
    mc_az = az_to_phi(mc_az * u.rad).value
    mc_alt_tel = alt_to_theta(mc_alt_tel * u.rad).value
    mc_az_tel = az_to_phi(mc_az_tel * u.rad).value

    # Sines and cosines of direction angles
    cp = np.cos(mc_az)
    sp = np.sin(mc_az)
    ct = np.cos(mc_alt)
    st = np.sin(mc_alt)

    # Shower direction coordinates

    sourcex = st * cp
    sourcey = st * sp
    sourcez = ct

    # print(sourcex)

    source = np.array([sourcex, sourcey, sourcez])
    source = source.T

    # Rotation matrices towars the camera frame

    rot_Matrix = np.empty((0, 3, 3))

    alttel = mc_alt_tel
    aztel = mc_az_tel
    mat_Y = np.array([[np.cos(alttel), 0, np.sin(alttel)],
                      [0, 1, 0],
                      [-np.sin(alttel), 0, np.cos(alttel)]], dtype=float).T

    mat_Z = np.array([[np.cos(aztel), -np.sin(aztel), 0],
                      [np.sin(aztel), np.cos(aztel), 0],
                      [0, 0, 1]], dtype=float).T

    rot_Matrix = np.matmul(mat_Y, mat_Z)

    res = np.einsum("...ji,...i", rot_Matrix, source)
    res = res.T

    source_x = -focal_length * res[0] / res[2]
    source_y = -focal_length * res[1] / res[2]
    return -source_y, -source_x


def get_event_pos_in_camera(event, tel):
    """
    Return the position of the source in the camera frame
    Parameters
    ----------
    event: `ctapipe.io.containers.DataContainer`
    tel: `ctapipe.instruement.telescope.TelescopeDescription`
    Returns
    -------
    (x, y) (float, float): position in the camera
    """
    array_pointing = SkyCoord(alt=event.mcheader.run_array_direction[1],
                              az=event.mcheader.run_array_direction[0],
                              frame=horizon_frame)

    event_direction = SkyCoord(alt=event.mc.alt,
                               az=event.mc.az,
                               frame=horizon_frame)

    focal = tel.optics.equivalent_focal_length

    camera_frame = CameraFrame(focal_length=focal,
                               telescope_pointing=array_pointing)

    camera_pos = event_direction.transform_to(camera_frame)
    return camera_pos.x, camera_pos.y


def reco_source_position_sky(cog_x, cog_y, disp_dx, disp_dy, focal_length,
                             pointing_alt, pointing_az):
    """
    Compute the reconstructed source position in the sky
    Parameters
    ----------
    cog_x: `astropy.units.Quantity`
    cog_y: `astropy.units.Quantity`
    disp_dx: `astropy.units.Quantity`
    disp_dy: `astropy.units.Quantity`
    focal_length: `astropy.units.Quantity`
    pointing_alt: `astropy.units.Quantity`
    pointing_az: `astropy.units.Quantity`
    Returns
    -------
    sky frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """
    src_x, src_y = disp.disp_to_pos(disp_dx, disp_dy, cog_x, cog_y)
    return camera_to_sky(src_x, src_y, focal_length, pointing_alt, pointing_az)


def camera_to_sky(pos_x, pos_y, focal, pointing_alt, pointing_az):
    """
    Parameters
    ----------
    pos_x: X coordinate in camera (distance)
    pos_y: Y coordinate in camera (distance)
    focal: telescope focal (distance)
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit
    Returns
    -------
    sky frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    Example:
    --------
    import astropy.units as u
    import numpy as np
    pos_x = np.array([0, 0]) * u.m
    pos_y = np.array([0, 0]) * u.m
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.camera_to_sky(pos_x, pos_y, focal, pointing_alt, pointing_az)
    """
    pointing_direction = SkyCoord(alt=pointing_alt, az=pointing_az,
                                  frame=horizon_frame)

    camera_frame = CameraFrame(focal_length=focal,
                               telescope_pointing=pointing_direction)

    camera_coord = SkyCoord(pos_x, pos_y, frame=camera_frame)

    horizon = camera_coord.transform_to(horizon_frame)

    return horizon


def sky_to_camera(alt, az, focal, pointing_alt, pointing_az):
    """
    Coordinate transform from aky position (alt, az) (in angles) to camera coordinates (x, y) in distance
    Parameters
    ----------
    alt: astropy Quantity
    az: astropy Quantity
    focal: astropy Quantity
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit
    Returns
    -------
    camera frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """
    pointing_direction = SkyCoord(alt=pointing_alt, az=pointing_az,
                                  frame=horizon_frame)

    camera_frame = CameraFrame(focal_length=focal,
                               pointing_direction=pointing_direction)

    event_direction = SkyCoord(alt=alt, az=az, frame=horizon_frame)

    camera_pos = event_direction.transform_to(camera_frame)

    return camera_pos


def source_side(source_pos_x, cog_x):
    """
    Compute on what side of the center of gravity the source is in the camera.
    Parameters
    ----------
    source_pos_x: X coordinate of the source in the camera, float
    cog_x: X coordinate of the center of gravity, float
    Returns
    -------
    float: -1 or +1
    """
    return np.sign(source_pos_x - cog_x)


def source_dx_dy(source_pos_x, source_pos_y, cog_x, cog_y):
    """
    Compute the coordinates of the vector (dx, dy) from the center of gravity to the source position
    Parameters
    ----------
    source_pos_x: X coordinate of the source in the camera
    source_pos_y: Y coordinate of the source in the camera
    cog_x: X coordinate of the center of gravity in the camera
    cog_y: Y coordinate of the center of gravity in the camera
    Returns
    -------
    (dx, dy)
    """
    return source_pos_x - cog_x, source_pos_y - cog_y


def polar_to_cartesian(norm, angle, sign):
    """
    Polar to cartesian transformation.
    As a convention, angle should be in [-pi/2:pi/2].
    Parameters
    ----------
    norm: float or `numpy.ndarray`
    angle: float or `numpy.ndarray`
    sign: float or `numpy.ndarray`
    Returns
    -------
    """
    assert np.isfinite([norm, angle, sign]).all()
    x = norm * sign * np.cos(angle)
    y = norm * sign * np.sin(angle)
    return x, y


def cartesian_to_polar(x, y):
    """
    Cartesian to polar transformation
    As a convention, angle is always included in [-pi/2:pi/2].
    When the angle should be in [pi/2:3*pi/2], angle = -1
    Parameters
    ----------
    x: float or `numpy.ndarray`
    y: float or `numpy.ndarray`
    Returns
    -------
    norm, angle, sign
    """
    norm = np.sqrt(x ** 2 + y ** 2)
    if x == 0:
        angle = np.pi / 2. * np.sign(y)
    else:
        angle = np.arctan(y / x)
    sign = np.sign(x)
    return norm, angle, sign


def predict_source_position_in_camera(cog_x, cog_y, disp_dx, disp_dy):
    """
    Compute the source position in the camera frame
    Parameters
    ----------
    cog_x: float or `numpy.ndarray` - x coordinate of the center of gravity (hillas.x)
    cog_y: float or `numpy.ndarray` - y coordinate of the center of gravity (hillas.y)
    disp_dx: float or `numpy.ndarray`
    disp_dy: float or `numpy.ndarray`
    Returns
    -------
    source_pos_x, source_pos_y
    """
    reco_src_x = cog_x + disp_dx
    reco_src_y = cog_y + disp_dy
    return reco_src_x, reco_src_y


def expand_tel_list(tel_list, max_tels):
    """
    transform for the telescope list (to turn it into a telescope pattern)
    un-pack var-length list of tel_ids into
    fixed-width bit pattern by tel_index
    """
    pattern = np.zeros(max_tels).astype(bool)
    pattern[tel_list] = 1
    return pattern


def filter_events(events,
                  filters=dict(intensity=[0, np.inf],
                               width=[0, np.inf],
                               length=[0, np.inf],
                               wl=[0, np.inf],
                               r=[0, np.inf],
                               leakage=[0, 1],
                               ),
                  dropna=True,
                  ):
    """
    Apply data filtering to a pandas dataframe.
    Each filtering range is applied if the column name exists in the DataFrame so that
    `(events >= range[0]) & (events <= range[1])`
    If the column name does not exist, the filtering is simply not applied
    Parameters
    ----------
    events: `pandas.DataFrame`
    filters: dict containing events features names and their filtering range
    dropna: bool
        if True (default), `dropna()` is applied to the dataframe.
    Returns
    -------
    `pandas.DataFrame`
    """

    filter = np.ones(len(events), dtype=bool)

    for k in filters.keys():
        if k in events.columns:
            filter = filter & (events[k] >= filters[k][0]) & (
                        events[k] <= filters[k][1])

    if dropna:
        return events[filter].dropna()
    else:
        return events[filter]


def linear_imputer(y, missing_values=np.nan, copy=True):
    """
    Replace missing values in y with values from a linear interpolation on their position in the array.
    Parameters
    ----------
    y: list or `numpy.array`
    missing_values: number, string, np.nan or None, default=`np.nan`
        The placeholder for the missing values. All occurrences of `missing_values` will be imputed.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will be done in-place whenever possible.
    Returns
    -------
    `numpy.array` : array with `missing_values` imputed
    """
    x = np.arange(len(y))
    if missing_values is np.nan:
        mask_missing = np.isnan(y)
    else:
        mask_missing = y == missing_values
    imputed_values = np.interp(x[mask_missing], x[~mask_missing],
                               y[~mask_missing])
    if copy:
        yy = np.copy(y)
        yy[mask_missing] = imputed_values
        return yy
    else:
        y[mask_missing] = imputed_values
        return y


def impute_pointing(dl1_data, missing_values=np.nan):
    """
    Impute missing pointing values using `linear_imputer` and replace them inplace
    Parameters
    ----------
    dl1_data: `pandas.DataFrame`
    missing_values: number, string, np.nan or None, default=`np.nan`
        The placeholder for the missing values. All occurrences of `missing_values` will be imputed.
    """
    if len(set(dl1_data.event_id)) != len(dl1_data.event_id):
        warn(
            "Beware, the data has been resorted by `event_id` to interpolate invalid pointing values but there are "
            "several events with the same `event_id` in the data, thus probably leading to unexpected behaviour",
            UserWarning)
    dl1_data = dl1_data.sort_values(by='event_id')
    for k in ['alt_tel', 'az_tel']:
        dl1_data[k] = linear_imputer(dl1_data[k].values,
                                     missing_values=missing_values)
    return dl1_data

class DispContainer(Container):
    """
    Disp vector container
    """
    dx = Field(np.nan, 'x coordinate of the disp_norm vector')
    dy = Field(np.nan, 'y coordinate of the disp_norm vector')

    angle = Field(np.nan, 'Angle between the X axis and the disp_norm vector')
    norm = Field(np.nan, 'Norm of the disp_norm vector')
    sign = Field(np.nan, 'Sign of the disp_norm')
    miss = Field(np.nan, 'miss parameter norm')


def disp(cog_x, cog_y, src_x, src_y):
    """
    Compute the disp parameters

    Parameters
    ----------
    cog_x: `numpy.ndarray` or float
    cog_y: `numpy.ndarray` or float
    src_x: `numpy.ndarray` or float
    src_y: `numpy.ndarray` or float

    Returns
    -------
    (disp_dx, disp_dy, disp_norm, disp_angle, disp_sign):
        disp_dx: 'astropy.units.m`
        disp_dy: 'astropy.units.m`
        disp_norm: 'astropy.units.m`
        disp_angle: 'astropy.units.rad`
        disp_sign: `numpy.ndarray`
    """
    disp_dx = src_x - cog_x
    disp_dy = src_y - cog_y
    disp_norm = np.sqrt(disp_dx**2 + disp_dy**2)
    if type(disp_dx) == float:
        if disp_dx == 0:
            disp_angle = np.pi/2. * np.sign(disp_dy)
        else:
            disp_angle = np.arctan(disp_dy/disp_dx)
    else:
        disp_angle = np.arctan(disp_dy / disp_dx)
        disp_angle[disp_dx==0] = np.pi/2. * disp_angle.unit * np.sign(disp_dy[disp_dx==0])

    disp_sign = np.sign(disp_dx)

    return disp_dx, disp_dy, disp_norm, disp_angle, disp_sign


def miss(disp_dx, disp_dy, hillas_psi):
    """
    Compute miss

    Parameters
    ----------
    disp_dx: `numpy.ndarray` or float
    disp_dy: `numpy.ndarray` or float
    hillas_psi: `numpy.ndarray` or float

    Returns
    -------

    """
    return np.abs(np.sin(hillas_psi) * disp_dx - np.cos(hillas_psi)*disp_dy)


def disp_parameters(cog_x, cog_y, mc_alt, mc_az, mc_alt_tel, mc_az_tel, focal):
    """
    Compute disp parameters.

    Parameters
    ----------
    cog_x: `numpy.ndarray` or float
    cog_y: `numpy.ndarray` or float
    mc_alt: `numpy.ndarray` or float
    mc_az: `numpy.ndarray` or float
    mc_alt_tel: `numpy.ndarray` or float
    mc_az_tel: `numpy.ndarray` or float
    focal: `numpy.ndarray` or float

    Returns
    -------
    (disp_dx, disp_dy, disp_norm, disp_angle, disp_sign) : `numpy.ndarray` or float
    """
    source_pos_in_camera = sky_to_camera(mc_alt, mc_az, focal, mc_alt_tel, mc_az_tel)
    return disp(cog_x, cog_y, source_pos_in_camera.x, source_pos_in_camera.y)



def disp_parameters_event(hillas_parameters, source_pos_x, source_pos_y):
    """
    Compute the disp_norm parameters from Hillas parameters in the event position in the camera frame
    Return a `DispContainer`

    Parameters
    ----------
    hillas_parameters: `ctapipe.io.containers.HillasParametersContainer`
    source_pos_x: `astropy.units.quantity.Quantity`
        X coordinate of the source (event) position in the camera frame
    source_pos_y: `astropy.units.quantity.Quantity`
        Y coordinate of the source (event) position in the camera frame

    Returns
    -------
    `lstchain.io.containers.DispContainer`
    """
    disp_container = DispContainer()

    d = disp(hillas_parameters.x.to(u.m).value,
             hillas_parameters.y.to(u.m).value,
             source_pos_x.to(u.m).value,
             source_pos_y.to(u.m).value,
             )

    disp_container.dx = d[0] * u.m
    disp_container.dy = d[1] * u.m
    disp_container.norm = d[2] * u.m
    disp_container.angle = d[3] * u.rad
    disp_container.sign = d[4]
    disp_container.miss = miss(disp_container.dx.value,
                               disp_container.dy.value,
                               hillas_parameters.psi.to(u.rad).value) * u.m
    return disp_container



def disp_vector(disp_norm, disp_angle, disp_sign):
    """
    Compute `disp_norm.dx` and `disp_norm.dy` vector from `disp_norm.norm`, `disp_norm.angle` and `disp_norm.sign`

    Parameters
    ----------
    disp_norm: float
    disp_angle: float
    disp_sign: float

    Returns
    -------
    disp_dx, disp_dy
    """
    return polar_to_cartesian(disp_norm, disp_angle, disp_sign)


def disp_to_pos(disp_dx, disp_dy, cog_x, cog_y):
    """
    Calculates source position in camera coordinates(x,y) from the reconstructed disp

    Parameters:
    -----------
    disp: DispContainer
    cog_x: float
    Coordinate x of the center of gravity of Hillas ellipse
    cog_y: float
    Coordinate y of the center of gravity of Hillas ellipse

    Returns:
    --------
    (source_pos_x, source_pos_y)
    """
    source_pos_x = cog_x + disp_dx
    source_pos_y = cog_y + disp_dy

    return source_pos_x, source_pos_y


def angular_distance(alt, az, tel_alt, tel_az):

    theta = np.arccos(np.sin(alt) * np.sin(tel_alt) + np.cos(alt) * np.cos(tel_alt) * np.cos(az - tel_az))
    return theta