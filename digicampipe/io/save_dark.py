import numpy as np
import itertools


def save_dark(data_stream, output_filename, n_events=None):

    if n_events is None:

        iter_events = itertools.count()

    else:

        iter_events = range(n_events)

    for event, i in zip(data_stream, iter_events):

        for telescope_id in event.r0.tels_with_data:

            if i == 0:

                data = event.r0.tel[telescope_id].adc_samples
                baseline = np.zeros(data.shape[0])
                std = np.zeros(data.shape[0])

            data = event.r0.tel[telescope_id].adc_samples
            baseline += np.mean(data, axis=-1)
            std += np.std(data, axis=-1)

        yield event

    baseline /= i
    std /= i
    np.savez(output_filename, baseline=baseline, standard_deviation=std)


