from ctapipe.image import hillas


def calibrate_dl1(event_stream):

    first = True

    for event in event_stream:

        if first:

            pixel_x, pixel_y = event.inst.pixel_pos
            first = False

        for telescope_id in event.dl0.tels_with_data:

            image = event.dl0.tel[telescope_id].pe_samples
            hillas_parameters = hillas.hillas_parameters_1(pixel_x, pixel_y, image)
            event.dl1.shower = hillas_parameters

        yield event
