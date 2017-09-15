from ctapipe.image import hillas


def calibrate_to_dl2(event_stream):

    first = True

    for event in event_stream:

        for telescope_id in event.dl2.tels_with_data:

            if first:
                pixel_x, pixel_y = event.inst.pixel_pos[telescope_id]
                first = False

            image = event.dl1.tel[telescope_id].image
            moments = hillas.hillas_parameters_1(pixel_x, pixel_y, image)
            event.dl2.shower = moments

            """
            
            event.dl1.energy = None
            event.dl1.classification = None

            """

        yield event
