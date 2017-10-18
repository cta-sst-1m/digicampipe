import numpy as np
import itertools


def save_bias_curve(data_stream, thresholds, output_filename, camera, n_events=None, blinding=True, by_cluster=True, unwanted_cluster=None):

    if n_events is None:

        iter_events = itertools.count()

    else:

        iter_events = range(n_events)

    rate = np.zeros(thresholds.shape)
    n_clusters = len(camera.Clusters_7)
    cluster_rate = np.zeros((n_clusters, thresholds.shape[-1]))
    cluster_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

    for cluster in camera.Clusters_7:

        for patch in cluster.patchesID:

            cluster_matrix[cluster.ID, patch] = 1

    if unwanted_cluster is not None:
        cluster_matrix[unwanted_cluster] = 0

    for event, i in zip(data_stream, iter_events):

        for telescope_id in event.r0.tels_with_data:

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces

            trigger_input_patch = np.dot(cluster_matrix, trigger_in)

            for j, threshold in enumerate(reversed(thresholds)):

                comp = trigger_input_patch > threshold

                if blinding:

                    if np.any(comp):

                        if by_cluster:
                            index = np.where(comp)[0]
                            cluster_rate[index, thresholds.shape[0] - j - 1] += 1
                            rate[thresholds.shape[0] - j - 1] += 1

                        else:

                            rate[0:thresholds.shape[0] - j] += 1
                            break

                else:
                    comp = np.sum(comp, axis=0)
                    comp = comp > 0
                    n_triggers = np.sum(comp)

                    if n_triggers > trigger_input_patch.shape[-1] - 1:

                        rate[0:thresholds.shape[0] - j] += n_triggers
                        break

                    rate[thresholds.shape[0] - j - 1] += n_triggers

            yield event

    time = ((i + 1) * 4. * trigger_input_patch.shape[-1])
    rate_error = np.sqrt(rate) / time
    cluster_rate_error = np.sqrt(cluster_rate) / time
    rate = rate / time
    cluster_rate = cluster_rate / time

    np.savez(output_filename, rate=rate, rate_error=rate_error, cluster_rate=cluster_rate, cluster_rate_error=cluster_rate_error, threshold=thresholds)


