import numpy as np


def fill_trigger_patch(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            data = event.r0.tel[telescope_id].adc_samples - event.r0.tel[telescope_id].baseline[:, np.newaxis].astype(int)
            data = np.dot(event.inst.patch_matrix[telescope_id], data)
            event.r0.tel[telescope_id].trigger_input_traces = data

        yield event


def fill_trigger_input_7(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_7 = np.dot(event.inst.cluster_matrix_7[telescope_id], trigger_in)
            event.r0.tel[telescope_id].trigger_input_7 = trigger_input_7

        yield event


def fill_trigger_input_19(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_19 = np.dot(event.inst.cluster_matrix_19[telescope_id], trigger_in)
            event.r0.tel[telescope_id].trigger_input_19 = trigger_input_19

        yield event


def fill_event_type(event_stream, flag):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            event.r0.tel[telescope_id].event_type_1 = flag
            event.r0.tel[telescope_id].event_type_2 = flag

        yield event
