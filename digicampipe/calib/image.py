from ctapipe.image import hillas_parameters


def compute_hillas_parameters(events, geom):
    for event in events:
        mask = event.data.cleaning_mask
        image = event.data.reconstructed_number_of_pe
        image[~mask] = 0
        hillas = hillas_parameters(geom, image)
        event.hillas = hillas

        yield event
