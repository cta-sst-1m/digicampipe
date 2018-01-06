# Functions for extracting and saving cleaned events, called from dl2.py

import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
import astropy.units as u
from ctapipe.instrument import CameraGeometry
import matplotlib.pyplot as plt


def make_image(geom: CameraGeometry, image, container=False):
    
    pix_x = Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(geom.pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
        
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape
        
    return pix_x, pix_y, image
    
    
def save_image(pix_x, pix_y, image):

    np.savetxt('pixels.txt',np.vstack((pix_x,pix_y)),'%1.4f')
    np.savetxt('events_image.txt',image,'%1.5f')


def load_image(pixels_file,events_file):
    pixels = np.loadtxt(pixels_file)
    events = np.loadtxt(events_file)
    
    return pixels, events
    
