import numpy as np
from digicampipe.instrument.camera import DigiCam


def fill_digicam_baseline(event_stream):

    for count, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            baseline = event.r0.tel[telescope_id].digicam_baseline
            event.r0.tel[telescope_id].baseline = baseline

        yield event


def compute_trigger_patch(adc_samples, baseline,
                          patch_matrix=DigiCam.patch_matrix):

    # baseline = np.floor(baseline)
    # trigger_patch = patch_matrix.dot(adc_samples)
    # baseline = patch_matrix.dot(baseline)
    # trigger_patch = trigger_patch - baseline[:, np.newaxis]
    # trigger_patch = np.clip(trigger_patch, 0, 255)
    # trigger_patch = trigger_patch.astype(np.uint16)

    # This allows to have negative integers and flooring of the baseline
    baseline = baseline.astype(int)
    adc_samples = adc_samples.astype(int)

    adc_samples = adc_samples - baseline[:, np.newaxis]
    trigger_patch = patch_matrix.dot(adc_samples)
    trigger_patch = np.clip(trigger_patch, 0, 255)

    return trigger_patch


def compute_trigger_input_7(trigger_patch,
                            cluster_matrix=DigiCam.cluster_7_matrix):

    trigger_input_7 = cluster_matrix.dot(trigger_patch)
    trigger_input_7 = np.clip(trigger_input_7, 0, 1785)

    return trigger_input_7


def compute_trigger_output_7(trigger_input_7, threshold):

    return trigger_input_7 > threshold


def fill_trigger_patch(event_stream):

    for count, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            matrix_patch = event.inst.patch_matrix[telescope_id]
            data = event.r0.tel[telescope_id].adc_samples
            baseline = event.r0.tel[telescope_id].baseline
            data = compute_trigger_patch(data, baseline, matrix_patch)
            event.r0.tel[telescope_id].trigger_input_traces = data

        yield event


def fill_trigger_input_7(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            matrix_patch_7 = event.inst.cluster_matrix_7[telescope_id]

            trigger_patch = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_7 = compute_trigger_input_7(trigger_patch,
                                                      matrix_patch_7)
            event.r0.tel[telescope_id].trigger_input_7 = trigger_input_7

        yield event


def fill_trigger_input_19(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            matrix_19 = event.inst.cluster_matrix_19[telescope_id]

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_19 = matrix_19.dot(trigger_in)
            trigger_input_19 = np.clip(trigger_input_19, 0, 1785)
            event.r0.tel[telescope_id].trigger_input_19 = trigger_input_19

        yield event


def fill_trigger_output_patch_19(event_stream, threshold):

    for event in event_stream:

        for tel_id, r0_container in event.r0.tel.items():

            trigger_input_19 = r0_container.trigger_input_19
            r0_container.trigger_output_patch_19 = trigger_input_19 > threshold

        yield event


def fill_trigger_output_patch_7(event_stream, threshold):

    for event in event_stream:

        for tel_id, r0_container in event.r0.tel.items():

            trigger_input_7 = r0_container.trigger_input_7
            out = compute_trigger_output_7(trigger_input_7, threshold)
            r0_container.trigger_output_patch_7 = out

        yield event


def fill_event_type(event_stream, flag):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            event.r0.tel[telescope_id].camera_event_type = flag

        yield event


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