from ctapipe.image import hillas_parameters, HillasParameterizationError
import numpy as np


def compute_hillas_parameters(events, geom):

    for event in events:
        mask = event.data.cleaning_mask
        image = event.data.reconstructed_number_of_pe
        image = np.ma.masked(image, mask=mask)

        try:

            hillas = hillas_parameters(geom, image)
            event.hillas = hillas

            yield event

        except HillasParameterizationError:

            continue
