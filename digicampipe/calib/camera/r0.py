import numpy as np
from digicampipe.utils import DigiCam

def fill_digicam_baseline(event_stream):

    for count, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            baseline = event.r0.tel[telescope_id].digicam_baseline
            event.r0.tel[telescope_id].baseline = baseline

        yield event


def compute_trigger_patch(adc_samples, baseline,
                          patch_matrix=DigiCam.patch_matrix):

    baseline = np.floor(baseline)
    baseline = np.clip(baseline, 0, 4095)
    baseline = baseline.astype(np.uint16)

    adc_samples = adc_samples - baseline[:, np.newaxis]
    adc_samples = np.clip(adc_samples, 0, 4095)

    trigger_patch = patch_matrix.dot(adc_samples)
    trigger_patch = np.clip(trigger_patch, 0, 255)
    trigger_patch = trigger_patch.astype(np.uint16)

    return trigger_patch


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

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_7 = matrix_patch_7.dot(trigger_in)
            trigger_input_7 = np.clip(trigger_input_7, 0, 255)
            event.r0.tel[telescope_id].trigger_input_7 = trigger_input_7

        yield event


def fill_trigger_input_19(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            matrix_19 = event.inst.cluster_matrix_19[telescope_id]

            trigger_in = event.r0.tel[telescope_id].trigger_input_traces
            trigger_input_19 = matrix_19.dot(trigger_in)
            trigger_input_19 = np.clip(trigger_input_19, 0, 255)
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
            r0_container.trigger_output_patch_7 = trigger_input_7 > threshold

        yield event


def fill_event_type(event_stream, flag):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            event.r0.tel[telescope_id].camera_event_type = flag

        yield event
