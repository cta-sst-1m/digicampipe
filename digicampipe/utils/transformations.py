import numpy as np
from astroquery.vizier import Vizier
from ctapipe.coordinates import CameraFrame, HorizonFrame
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy import units as u

ifj_krakow = EarthLocation(
    lat=50.090815 * u.deg,
    lon= 19.887937 * u.deg,
    height=214.034 * u.m
)


def transform_azel_to_xy(stars_az, stars_alt, az_obs, el_obs):
    """
    function to transform azimuth and elevation coodinates to XY coordinates in
    the fild of view.
    :param stars_az: 2D array of azimuth coordinates to be transformed.
    1st dimension is for different objects and 2nd dimension is for different
    observation time.
    :param stars_alt: 2D array of elevation coordinates to be transformed.
    1st dimension is for different objects and 2nd dimension is for different
    observation time.
    :param az_obs: 1D array of telescope azimuth pointing direction (1 element
    per observation time).
    :param el_obs: 1D array of telescope elevation pointing direction (1
    element per observation time).
    :return: stars_x, stars_y : 2D array giving for each object (1st dimension)
    and each observation time (2nd dimension) the XY coordinates in the field
    of view.
    """
    n_stars = stars_az.shape[0]
    n_event = az_obs.shape[0]
    assert stars_az.shape == (n_stars, n_event)
    assert stars_alt.shape == (n_stars, n_event)
    assert el_obs.shape[0] == n_event
    stars_x = np.zeros([n_stars, n_event]) * u.mm
    stars_y = np.zeros([n_stars, n_event]) * u.mm
    for event in range(n_event):
        pd = SkyCoord(
            alt=el_obs[event],
            az=az_obs[event],
            frame=HorizonFrame()
        )
        cam_frame = CameraFrame(
            focal_length=5.6 * u.m,
            rotation=90 * u.deg,
            pointing_direction=pd,
            array_direction=pd,
        )
        stars_sky = SkyCoord(
            alt=stars_alt[:, event],
            az=stars_az[:, event],
            frame=HorizonFrame()
        )
        stars_cam = stars_sky.transform_to(cam_frame)
        stars_x[:, event] = -stars_cam.x
        stars_y[:, event] = stars_cam.y
    return stars_x, stars_y


def get_stars_in_fov(
        az_obs, el_obs, time_obs, site_location=ifj_krakow,
        radius=Angle(10, "deg"), catalog='I/254', Pmag_max=6):
    """
    function to get from a catalog the coordinates of the stars in the field
    of view.
    :param az_obs: azimuth coordinate where the telescope is pointing at.
    :param el_obs: elevation coordinate where the telescope is pointing at.
    :param time_obs: astropy.time.Time object containing the observation times.
    :param site_location: astropy.coordinates.EarthLocation describing the
    location of the telescope
    :param radius: coordinates of star closer than this radius around the
    pointing direction are returned, others are ignored.
    WARNING: for performance reason, only the first element of time_obs is used
    to obtain the stars. use a radius large enough if stars are drifting in the
    FoV, or call this function several time is this is problematic.
    :param catalog: Vizier catalog to use to obtain the stars.
    :param Pmag_max: stars with a magnitude smaller than Pmag_max will be
    ignored.
    :return: stars_az, stars_alt, stars_pmag. stars_az, stars_alt are arrays
    containing for each stars (1st dimension) and each time_obs (2nd dimension)
    the azimuth and elevation coordinates. stars_pmag contains the apparent
    magnitude of each star.
    """
    time_obs = np.atleast_1d(time_obs)
    n_event = len(time_obs)
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000', 'Pmag', ' Bmag'],
        column_filters={"Pmag": "<{}".format(Pmag_max)},
        row_limit=-1,
    )
    stars_table = vizier.query_region(
        AltAz(
            alt=el_obs,
            az=az_obs,
            obstime=time_obs[0],
            location=site_location
        ),
        radius=radius,
        catalog=catalog
    )[0]
    stars_pmag = stars_table['Pmag']
    n_star = len(stars_pmag)
    stars_alt = np.zeros([n_star, n_event]) * u.deg
    stars_az = np.zeros([n_star, n_event]) * u.deg
    for star_idx in range(n_star):
        skycoord = SkyCoord(
            ra=stars_table['RAJ2000'][star_idx] * u.deg,
            dec=stars_table['DEJ2000'][star_idx] * u.deg,
            obstime='J2000'
        )
        star_pos = skycoord.transform_to(
            AltAz(obstime=time_obs, location=site_location)
        )
        stars_alt[star_idx, :] = star_pos.alt
        stars_az[star_idx, :] = star_pos.az
    return stars_az, stars_alt, stars_pmag
