from ctapipe.image import hillas
import numpy as np
import astropy.units as u


def calibrate_to_dl2(event_stream, reclean=False, shower_distance=80*u.mm):

    for i, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            if i == 0:

                geom = event.inst.geom[telescope_id]
                pixel_x, pixel_y = geom.pix_x, geom.pix_y

            dl1_camera = event.dl1.tel[telescope_id]

            image = dl1_camera.pe_samples

            mask = dl1_camera.cleaning_mask
            image[~mask] = 0.

            try:

                moments_first = hillas.hillas_parameters(pixel_x, pixel_y, image)

                if reclean:

                    mask_near_center = find_mask_near_center(geom=geom,
                                                             cen_x=moments_first.cen_x,
                                                             cen_y=moments_first.cen_y,
                                                             distance=shower_distance)
                    dl1_camera.cleaning_mask = dl1_camera.cleaning_mask & mask_near_center
                    image[~dl1_camera.cleaning_mask] = 0
                    moments = hillas.hillas_parameters(pixel_x, pixel_y, image)
                else:
                    moments = moments_first
            except:

                print('could not recompute Hillas of event')
                moments = None
        event.dl2.shower = moments
        event.dl2.energy = None
        event.dl2.classification = None

        if moments is not None:
            # print(moments)
            yield event


def find_mask_near_center(geom, cen_x, cen_y, distance):

    d = np.sqrt((geom.pix_x - cen_x)**2 + (geom.pix_y - cen_y)**2)

    return d < distance


def find_mask_near_max(geom, distance, index_max):

    cen_x, cen_y = geom.pix_x[index_max], geom.pix_y[index_max]

    return find_mask_near_center(geom=geom, cen_x=cen_x, cen_y=cen_y, distance=distance)