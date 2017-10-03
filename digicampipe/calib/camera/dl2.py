from ctapipe.image import hillas
import numpy as np
import astropy.units as u
import cts_core.camera as camera
import digicampipe.utils.geometry as geometry


def calibrate_to_dl2(event_stream, reclean=False, camera_config_file=None, shower_distance=80*u.mm):

    if reclean:
        cam = camera.Camera(_config_file=camera_config_file)
        geom = geometry.generate_geometry(camera=cam)

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            dl1_camera = event.dl1.tel[telescope_id]

            pixel_x, pixel_y = event.inst.pixel_pos[telescope_id]
            image = dl1_camera.pe_samples

            mask = dl1_camera.cleaning_mask
            image[~mask] = 0.

            """
            for i,pe in enumerate(image):
                print(pixel_x[i], pixel_y[i], image[i])

            print(np.sum(np.ones(image.shape)[mask]))
            """

            try:

                moments_first = hillas.hillas_parameters_4(pixel_x, pixel_y, image)

                if reclean:

                    mask_near_center = find_mask_near_center(geom=geom, cen_x=moments_first.cen_x,
                                                             cen_y=moments_first.cen_y, distance=shower_distance)
                    dl1_camera.cleaning_mask = dl1_camera.cleaning_mask & mask_near_center
                    image[~dl1_camera.cleaning_mask] = 0
                    moments = hillas.hillas_parameters_4(pixel_x, pixel_y, image)
            except:

                print('could not recompute Hillas, not yielding')
                moments = None

        event.dl2.shower = moments
        event.dl2.energy = None
        event.dl2.classification = None

        if moments is not None:
            print(moments)
            yield event


def find_mask_near_center(geom, cen_x, cen_y, distance):

    d = np.sqrt((geom.pix_x - cen_x)**2 + (geom.pix_y - cen_y)**2)

    return d < distance


def find_mask_near_max(geom, distance, index_max):

    cen_x, cen_y = geom.pix_x[index_max], geom.pix_y[index_max]

    return find_mask_near_center(geom=geom, cen_x=cen_x, cen_y=cen_y, distance=distance)