import numpy as np


def filter_patch(event_stream, unwanted_patch):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

            patch_condition = np.any(output_trigger_patch7[unwanted_patch])

            if not patch_condition:
                # Set the event type
                event.trig.trigger_flag = 0
                event.level = 0
                yield event
            else:
                # Set the event type
                event.trig.trigger_flag = 1
                event.level = 0
                yield event


def filter_level(event_stream, level = 1):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param level:
    :return:
    """
    for event in event_stream:
        if event.level >= level :
            yield event


def filter_bigshower(event_stream, minpe = 10000):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param level:
    :return:
    """
    for event in event_stream:
        r1 = event.r1.tel[event.r0.tels_with_data[0]]
        if np.sum(r1.pe_samples[r1.cleaning_mask]) >= minpe :
            yield event



def filter_trigger_time(event_stream, time):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

            print(np.sum(output_trigger_patch7))
            condition = np.sum(output_trigger_patch7) > time

            if condition:

                yield event
