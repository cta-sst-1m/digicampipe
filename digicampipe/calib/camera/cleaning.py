import numpy as np
from ctapipe.image import cleaning

def compute_cleaning_1(events, snr=3, overwrite=True):

    for event in events:

        baseline_std = event.data.baseline_std
        max_amplitude = np.max(event.data.adc_samples, axis=-1)
        mask = max_amplitude > (snr * baseline_std)

        if overwrite:

            event.data.cleaning_mask = mask

        else:
            event.data.cleaning_mask *= mask

        yield event


def compute_tailcuts_clean(events, geom, overwrite=True, **kwargs):

    for event in events:

        image = event.data.reconstructed_number_of_pe
        mask = cleaning.tailcuts_clean(geom=geom, image=image, **kwargs)

        if overwrite:

            event.data.cleaning_mask = mask

        else:
            event.data.cleaning_mask *= mask

        yield event


def compute_boarder_cleaning(events, geom, boundary_threshold):

    pixel_id = np.array(geom.pix_id)

    for event in events:

        mask = event.data.cleaning_mask
        if not np.any(mask):

            continue

        image = event.data.reconstructed_number_of_pe
        recursion = True
        while recursion:

            recursion = False

            for i in pixel_id[mask]:

                for j in pixel_id[geom.neighbor_matrix[i] & (~mask)]:

                    if image[j] > boundary_threshold:

                        mask[j] = True
                        recursion = True

        event.data.cleaning_mask = mask

        num_neighbors = np.sum(geom.neighbor_matrix[mask], axis=-1)
        on_border = np.any(num_neighbors < 6)

        if not on_border:

            yield event


def compute_dilate(events, geom):

    for event in events:

        mask = event.data.cleaning_mask
        mask = cleaning.dilate(geom, mask)
        event.data.cleaning_mask = mask

        yield event
