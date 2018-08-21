from ctapipe.image import hillas
import numpy as np
import astropy.units as u


def hillas_parameters(geom, image):
    try:
        return hillas.hillas_parameters(geom, image)
    except hillas.HillasParameterizationError:
        unit = geom.pix_x.unit
        return hillas.MomentParameters(
            intensity=0.,
            x=np.nan * unit,
            y=np.nan * unit,
            length=np.nan * unit,
            width=np.nan * unit,
            r=np.nan * unit,
            phi=hillas.Angle(np.nan * u.rad),
            psi=hillas.Angle(np.nan * u.rad),
            # miss=np.nan * unit,
            skewness=None,
            kurtosis=None,
        )


def calibrate_to_dl2(event_stream, reclean=False, shower_distance=80*u.mm):
    '''
    Skips events with intensity==0
    '''

    for i, event in enumerate(event_stream):

        moments = None
        for telescope_id in event.r0.tels_with_data:

            if i == 0:

                geom = event.inst.geom[telescope_id]

            dl1_camera = event.dl1.tel[telescope_id]

            image = dl1_camera.pe_samples

            mask = dl1_camera.cleaning_mask
            image[~mask] = 0.
            moments_first = hillas_parameters(geom, image)
            if moments_first.intensity == 0:
                continue
            if reclean:
                mask_near_center = find_mask_near_center(
                    geom=geom,
                    cen_x=moments_first.x,
                    cen_y=moments_first.y,
                    distance=shower_distance)
                dl1_camera.cleaning_mask &= mask_near_center
                image[~dl1_camera.cleaning_mask] = 0
                moments = hillas_parameters(geom, image)
                if moments.intensity == 0:
                    continue
            else:
                moments = moments_first
        if moments is None:
            continue
        event.dl2.shower = moments
        event.dl2.energy = None
        event.dl2.classification = None

        yield event


def find_mask_near_center(geom, cen_x, cen_y, distance):

    d = np.sqrt(
        (geom.pix_x - cen_x)**2 +
        (geom.pix_y - cen_y)**2
    )

    return d < distance


def find_mask_near_max(geom, distance, index_max):
    return find_mask_near_center(
        geom=geom,
        cen_x=geom.pix_x[index_max],
        cen_y=geom.pix_y[index_max],
        distance=distance
    )
