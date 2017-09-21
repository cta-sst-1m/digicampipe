from cts_core import camera as cam
import numpy as np
import astropy.units as u

def find_pixel_positions():

    camera_config_file = '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg'
    camera = cam.Camera(_config_file=camera_config_file)

    x, y = [], []

    for pixel in camera.Pixels:

        x.append(pixel.center[0])
        y.append(pixel.center[1])

    return np.array([x, y]) * u.mm
