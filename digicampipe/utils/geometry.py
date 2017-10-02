from cts_core import camera as cam
import numpy as np
import astropy.units as u
from digicampipe.instrument.camera import CameraGeometry
from ctapipe.io.camera import find_neighbor_pixels


def find_pixel_positions(camera_config_file):
    camera = cam.Camera(_config_file=camera_config_file)

    x, y = [], []

    for pixel in camera.Pixels:

        x.append(pixel.center[0])
        y.append(pixel.center[1])

    return np.array([x, y]) * u.mm


def generate_geometry(camera):
    """
    Generate the SST-1M geometry from the CTS configuration
    :param cts: a CTS instance
    :param available_board:  which board per sector are available (dict)
    :return: the geometry for visualisation and the list of "good" pixels
    """
    pix_x = []
    pix_y = []
    pix_id = []

    for pix in camera.Pixels:
        pix_x.append(pix.center[0])
        pix_y.append(pix.center[1])
        pix_id.append(pix.ID)

    neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
    geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones(1296) * 400., pix_type='hexagonal',
                          neighbors=neighbors_pix)

    return geom