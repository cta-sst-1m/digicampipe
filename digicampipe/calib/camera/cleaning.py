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


def compute_3d_cleaning(events, geom, threshold_pe_frac=20, threshold_time=2.1,
                        threshold_size=0.005, n_sample=50, sampling_time=4,
                        n_pixel=1296):
    samples = np.arange(0, n_sample * sampling_time, sampling_time)
    pix_x = np.tile(geom.pix_x.value[:, None], [1, n_sample])
    pix_y = np.tile(geom.pix_y.value[:, None], [1, n_sample])
    pix_t = np.tile(samples[None, :], [n_pixel, 1])
    for event in events:
        pe_frac = event.data.reconstructed_fraction_of_pe
        # ignore missing pixels
        pe_frac[~np.isfinite(pe_frac)] = 0
        # set to 0 the samples with fractional pe lower than threshold_pe_frac
        pe_frac[pe_frac < threshold_pe_frac] = 0
        sum_t = np.nansum(pe_frac, axis=1)
        mean_t = np.nansum(pe_frac * pix_t, axis=1) / sum_t
        mean_t_tiled = np.tile(mean_t[:, None], [1, n_sample])
        var_t = np.nansum(pe_frac * (pix_t - mean_t_tiled) ** 2 , axis=1) / \
                sum_t
        # set fractional pe to 0 for pixels with too short pulses
        pe_frac[np.sqrt(var_t) < threshold_time, :] = 0

        sum_pixel_pe = np.nansum(pe_frac, axis=0)
        mean_x = np.nansum(pe_frac * pix_x, axis=0) / sum_pixel_pe
        mean_y = np.nansum(pe_frac * pix_y, axis=0) / sum_pixel_pe
        mean_x_tiled = np.tile(mean_x[None, :], [n_pixel, 1])
        mean_y_tiled = np.tile(mean_y[None, :], [n_pixel, 1])
        var_x = np.nansum(pe_frac * (pix_x - mean_x_tiled) ** 2 , axis=0) / \
                sum_pixel_pe
        var_y = np.nansum(pe_frac * (pix_y - mean_y_tiled) ** 2 , axis=0) / \
                sum_pixel_pe
        shower = False
        selection = np.logical_and(
            np.isfinite(var_x + var_y),
            var_x + var_y > 0
        )
        if np.any(selection):
            std_xy = np.mean(np.sqrt(var_x + var_y)[selection])
            shower = std_xy > threshold_size
        event.data.shower = shower
        yield event
