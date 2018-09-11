import numpy as np
from astropy import units as u
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


def compute_boarder_cleaning(events, geom, boundary_threshold, skip=False):

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
        event.data.border = on_border

        if on_border and skip:

            continue

        else:

            yield event


def compute_dilate(events, geom):

    for event in events:

        mask = event.data.cleaning_mask
        mask = cleaning.dilate(geom, mask)
        event.data.cleaning_mask = mask

        yield event


def compute_3d_cleaning(events, geom, threshold_sample_pe=20,
                        threshold_time=2.1 * u.ns, threshold_size=0.005 * u.mm,
                        n_sample=50, sampling_time=4 * u.ns):
    samples = np.arange(
        0, n_sample * sampling_time.value, sampling_time.value
    ) * sampling_time.unit
    pix_x = geom.pix_x[:, None]
    pix_y = geom.pix_y[:, None]
    pix_t = samples[None, :]
    for event in events:
        sample_pe = event.data.sample_pe
        # ignore missing pixels
        sample_pe[~np.isfinite(sample_pe)] = 0
        # set to 0 the samples with sample_pe lower than threshold_sample_pe
        sample_pe[sample_pe < threshold_sample_pe] = 0
        sum_t = np.nansum(sample_pe, axis=1)
        mean_t = np.nansum(sample_pe * pix_t, axis=1) / sum_t
        var_t = np.nansum(
            sample_pe * (pix_t - mean_t[:, None]) ** 2, axis=1
        ) / sum_t
        # set fractional pe to 0 for pixels with too short pulses
        sample_pe[np.sqrt(var_t) < threshold_time, :] = 0

        sum_pixel_pe = np.nansum(sample_pe, axis=0)
        mean_x = np.nansum(sample_pe * pix_x, axis=0) / sum_pixel_pe
        mean_y = np.nansum(sample_pe * pix_y, axis=0) / sum_pixel_pe
        var_x = np.nansum(
            sample_pe * (pix_x - mean_x[None, :]) ** 2,
            axis=0
        ) / sum_pixel_pe
        var_y = np.nansum(
            sample_pe * (pix_y - mean_y[None, :]) ** 2,
            axis=0
        ) / sum_pixel_pe
        shower = False
        selection = np.logical_and(
            np.isfinite(var_x + var_y),
            (var_x + var_y) > 0
        )
        if np.any(selection):
            std_xy = np.mean(np.sqrt(var_x + var_y)[selection])
            shower = std_xy > threshold_size
        event.data.shower = shower
        yield event
