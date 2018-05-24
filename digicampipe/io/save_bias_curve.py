import numpy as np


def init_cluster_rate(r0, n_thresholds):
    n_clusters = r0.trigger_input_7.shape[0]
    cluster_rate = np.zeros((n_clusters, n_thresholds))
    return cluster_rate


def compute_bias_curve_v2(data_stream, thresholds):

    n_thresholds = len(thresholds)
    n_clusters = 432
    cluster_rate = np.zeros((n_clusters, n_thresholds))
    rate = np.zeros(n_thresholds)

    for count, event in enumerate(data_stream):

        for tel_id, r0 in event.r0.tel.items():

            trigger_input = r0.trigger_input_7

            comp = trigger_input[..., np.newaxis] > thresholds
            temp_cluster_rate = np.sum(comp, axis=1)
            temp_cluster_rate[temp_cluster_rate > 0] = 1
            cluster_rate += temp_cluster_rate

            temp_rate = np.sum(temp_cluster_rate, axis=0)
            temp_rate[temp_rate > 0] = 1
            rate += temp_rate

    time = ((count + 1) * 4. * trigger_input.shape[-1])
    rate_error = np.sqrt(rate) / time
    cluster_rate_error = np.sqrt(cluster_rate) / time
    rate = rate / time
    cluster_rate = cluster_rate / time

    return rate, rate_error, cluster_rate, cluster_rate_error, thresholds

oskofkaof
def compute_bias_curve(
    data_stream,
    thresholds,
    blinding=True,
    by_cluster=True,
):
    """
    :param data_stream:
    :param thresholds:
    :param blinding:
    :param by_cluster:
    :return:
    """
    n_thresholds = len(thresholds)
    rate = np.zeros(n_thresholds)

    # cluster rate can only be initialized when the first event was read.
    cluster_rate = None
    for event_id, event in enumerate(data_stream):

        for tel_id, r0 in event.r0.tel.items():

            if cluster_rate is None:
                cluster_rate = init_cluster_rate(r0, n_thresholds)

            for threshold_id, threshold in enumerate(reversed(thresholds)):

                comp = r0.trigger_input_7 > threshold

                if blinding:

                    if np.any(comp):

                        if by_cluster:
                            index = np.where(comp)[0]
                            cluster_rate[index, - threshold_id - 1] += 1
                            rate[- threshold_id - 1] += 1

                        else:

                            rate[0:-threshold_id] += 1
                            break

                else:
                    comp = np.sum(comp, axis=0)
                    comp = comp > 0
                    n_triggers = np.sum(comp)

                    if n_triggers > r0.trigger_input_7.shape[-1] - 1:

                        rate[0:-threshold_id] += n_triggers
                        break

                    rate[- threshold_id - 1] += n_triggers

    time = ((event_id + 1) * 4. * r0.trigger_input_7.shape[-1])
    rate_error = np.sqrt(rate) / time
    cluster_rate_error = np.sqrt(cluster_rate) / time
    rate = rate / time
    cluster_rate = cluster_rate / time

    return rate, rate_error, cluster_rate, cluster_rate_error, thresholds
