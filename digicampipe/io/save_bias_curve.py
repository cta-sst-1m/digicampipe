import numpy as np


def save_bias_curve(
    data_stream,
    thresholds,
    output_filename,
    n_events=None,
    blinding=True,
    by_cluster=True,
    unwanted_cluster=None
):
    rate = np.zeros(thresholds.shape)

    for event_id, event in enumerate(data_stream):
        if event_id >= n_events:
            break

        for telescope_id in event.r0.tels_with_data:

            trigger_input_7 = event.r0.tel[telescope_id].trigger_input_7

            if event_id == 0:

                n_clusters = trigger_input_7.shape[0]
                cluster_rate = np.zeros((n_clusters, thresholds.shape[-1]))

            for threshold_id, threshold in enumerate(reversed(thresholds)):

                comp = trigger_input_7 > threshold

                if blinding and np.any(comp):

                    if by_cluster:
                        index = np.where(comp)[0]
                        cluster_rate[index, thresholds.shape[0] - threshold_id - 1] += 1
                        rate[thresholds.shape[0] - threshold_id - 1] += 1

                    else:

                        rate[0:thresholds.shape[0] - threshold_id] += 1
                        break

                else:
                    comp = np.sum(comp, axis=0)
                    comp = comp > 0
                    n_triggers = np.sum(comp)

                    if n_triggers > trigger_input_7.shape[-1] - 1:

                        rate[0:thresholds.shape[0] - threshold_id] += n_triggers
                        break

                    rate[thresholds.shape[0] - threshold_id - 1] += n_triggers

        yield event

    time = ((event_id + 1) * 4. * trigger_input_7.shape[-1])
    rate_error = np.sqrt(rate) / time
    cluster_rate_error = np.sqrt(cluster_rate) / time
    rate = rate / time
    cluster_rate = cluster_rate / time

    np.savez(
        output_filename,
        rate=rate,
        rate_error=rate_error,
        cluster_rate=cluster_rate,
        cluster_rate_error=cluster_rate_error,
        threshold=thresholds
    )
