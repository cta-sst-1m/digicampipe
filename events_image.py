# Functions for extracting and saving cleaned events, called from pipeline_crab.py

import numpy as np
#from astropy.units import Quantity
from astropy.coordinates import Angle
import astropy.units as u
from ctapipe.instrument import CameraGeometry
import matplotlib.pyplot as plt


def make_image(geom: CameraGeometry, image, container=False):

    pix_x = np.asanyarray(geom.pix_x, dtype=np.float64).value       # Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = np.asanyarray(geom.pix_y, dtype=np.float64).value
    image = np.asanyarray(image, dtype=np.float64)

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    return pix_x, pix_y, image


def save_image(pix_x, pix_y, image, filename_pix, filename_eventsimage):

    np.savetxt(filename_pix, np.vstack((pix_x, pix_y)), '%1.4f')
    np.savetxt(filename_eventsimage, image, '%1.5f')


def load_image(pixels_file, events_file):

    pixels = np.loadtxt(pixels_file)
    events = np.loadtxt(events_file)

    return pixels, events


def save_events(event_stream, filename_pix, filename_eventsimage):

    image_all = []

    for i, event in enumerate(event_stream):
        for telescope_id in event.r0.tels_with_data:
            if i == 0:
                geom = event.inst.geom[telescope_id]
            dl1_camera = event.dl1.tel[telescope_id]
            image = dl1_camera.pe_samples
            mask = dl1_camera.cleaning_mask
            image[~mask] = 0.

        # saving cleaned event images
        pix_x, pix_y, image = make_image(geom, image)  #
        event_number = event.r0.event_id  #
        image = np.hstack((event_number, image))  # [event_ number, image_values]
        image_all.append(image)  #
        print('saving event',i)

        yield event

    save_image(pix_x, pix_y, image_all, filename_pix, filename_eventsimage)  # save cleaned images for all events


def save_timing(event_stream,filename_timing):

    timing_all = []

    for i, event in enumerate(event_stream):
        
        for telescope_id in event.r0.tels_with_data:

            dl1_camera = event.dl1.tel[telescope_id]
            mask = dl1_camera.cleaning_mask
            timing_data = dl1_camera.time_bin[1]
            timing_data[~mask] = 0.

        timing_all.append(timing_data)

        yield event

    np.savetxt(filename_timing, timing_all, '%1.5f')

