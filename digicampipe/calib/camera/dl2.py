from ctapipe.image import hillas


def calibrate_to_dl2(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            pixel_x, pixel_y = event.inst.pixel_pos[telescope_id]
            image = event.dl1.tel[telescope_id].pe_samples

            mask = event.dl1.tel[telescope_id].cleaning_mask

            image = image[mask]
            pixel_x = pixel_x[mask]
            pixel_y = pixel_y[mask]

            moments = hillas.hillas_parameters_2(pixel_x, pixel_y, image)

        event.dl2.shower = moments
        event.dl2.energy = None
        event.dl2.classification = None

        yield event
