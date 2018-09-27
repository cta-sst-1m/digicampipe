import astropy.units as u
import numpy as np


def set_patches_to_zero(event_stream, unwanted_patch):
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            if unwanted_patch is not None:
                r0_camera.trigger_input_traces[unwanted_patch] = 0
                r0_camera.trigger_output_patch7[unwanted_patch] = 0
                r0_camera.trigger_output_patch19[unwanted_patch] = 0

        yield event


def set_pixels_to_zero(event_stream, unwanted_pixels):
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            r0_camera.adc_samples[unwanted_pixels] = 0

            yield event


def filter_event_types(event_stream, flags=[0]):
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            flag = event.r0.tel[telescope_id].camera_event_type

            if flag in flags:
                yield event


def filter_shower(event_stream, min_photon):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param min_photon:
    :return:
    """
    for i, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:
            dl1_camera = event.dl1.tel[telescope_id]
            if np.sum(dl1_camera.pe_samples[
                          dl1_camera.cleaning_mask]) >= min_photon:
                yield event


def filter_shower_adc(event_stream, min_adc):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param min_adc:
    :return:
    """
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            r1_camera = event.r1.tel[telescope_id]
            if np.sum(np.max(r1_camera.adc_samples, axis=-1)) >= min_adc:
                yield event


def filter_missing_baseline(event_stream):
    for event in event_stream:
        for telescope_id in event.r0.tels_with_data:
            if event.r0.tel[telescope_id].baseline is not None:
                yield event


def filter_trigger_time(event_stream, time):
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = r0_camera.trigger_output_patch7

            condition = np.sum(output_trigger_patch7) > time

            if condition:
                yield event


def filter_period(event_stream, period):
    t_last = 0 * u.second

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            t_new = event.r0.tel[
                        telescope_id].local_camera_clock * u.nanosecond

            if (t_new - t_last) > period:
                t_last = t_new
                yield event


def filter_clocked_trigger(events):
    for event in events:

        if event.event_type.INTRNL not in event.event_type:
            yield event
