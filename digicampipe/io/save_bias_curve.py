import numpy as np
import itertools


def save_bias_curve(data_stream, output_filename, camera, n_events=None, unwanted_patch=None):

    if n_events is None:

        iter_events = itertools.count()

    else:

        iter_events = range(n_events)

    thresholds = np.arange(0, 800, 10)
    rate = np.zeros(thresholds.shape)
    n_samples = 50

    trigger_input_patch = np.zeros((432, n_samples))

    for event, i in zip(data_stream, iter_events):

        for telescope_id in event.r0.tels_with_data:

            trigger_in = np.array(list(event.r0.tel[telescope_id].trigger_input_traces.values()))

            for cluster in camera.Clusters_7:

                trigger_input_patch[cluster.ID] = np.sum(trigger_in[cluster.patchesID, :], axis=0)

            trigger_input_patch[unwanted_patch] = 0.

            for j, threshold in enumerate(reversed(thresholds)):

                if np.any(trigger_input_patch > threshold):

                    rate[0:thresholds.shape[0] - j+1] += 1
                    break

    time = ((i + 1) * 4 * n_samples)
    rate_error = np.sqrt(rate) / time
    rate = rate / time

    np.savez(output_filename, rate=rate, threshold=thresholds, rate_error=rate_error)


