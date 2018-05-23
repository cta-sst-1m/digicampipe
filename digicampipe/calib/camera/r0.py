import numpy as np


def fill_trigger_patch(event_stream):

    for count, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            matrix_patch = event.inst.patch_matrix[telescope_id]
            baseline = event.r0.tel[telescope_id].baseline[:, np.newaxis]
            baseline = np.floor(baseline)
            baseline = baseline.astype(int)

            data = event.r0.tel[telescope_id].adc_samples
            data = data - baseline
            data = matrix_patch.dot(data)
            event.r0.tel[telescope_id].trigger_input_traces = data

        yield event


def fill_trigger_input_7(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            matrix_patch_7 = event.inst.cluster_matrix_7[telescope_id]

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_7 = matrix_patch_7.dot(trigger_in)
            event.r0.tel[telescope_id].trigger_input_7 = trigger_input_7

        yield event


def fill_trigger_input_19(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            matrix_19 = event.inst.cluster_matrix_19[telescope_id]

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_19 = matrix_19.dot(trigger_in)
            event.r0.tel[telescope_id].trigger_input_19 = trigger_input_19

        yield event


def fill_event_type(event_stream, flag):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            event.r0.tel[telescope_id].camera_event_type = flag

        yield event
