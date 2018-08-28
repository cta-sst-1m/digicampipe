from ctapipe.image import hillas_parameters
from ctapipe import __version__ as ctapipe_version

def compute_hillas_parameters(events, geom):

    for event in events:

        mask = event.data.cleaning_mask
        image = event.data.reconstructed_number_of_pe
        image[~mask] = 0
        if ctapipe_version < '0.6':
            hillas = hillas_parameters(geom, image, container=True)
        else:
            hillas = hillas_parameters(geom, image)
        event.hillas = hillas

        yield event
