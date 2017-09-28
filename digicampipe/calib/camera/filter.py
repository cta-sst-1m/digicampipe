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
            else:
                # Set the event type
                event.trig.trigger_flag = 1

        yield event


def fill_flag(event_stream, unwanted_patch = None):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            if unwanted_patch is None:
                # Condition to be checked....
                print('event_type',r0_camera.event_type,'eventType',r0_camera.eventType)
                if r0_camera.event_type == 0:
                    # Physics
                    r0_camera.flag = 1
                else:
                    # Calib
                    r0_camera.flag = 0
            else:

                output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

                patch_condition = np.any(output_trigger_patch7[unwanted_patch])

                if not patch_condition:
                    # Set the event type
                    r0_camera.flag = 1
                else:
                    # Set the event type
                    r0_camera.flag = 0

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


def filter_flag(event_stream, flags=[0]):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            flag = event.r0.tel[telescope_id].flag

            if flag in flags:

                yield event


def filter_bigshower(event_stream, min_photon = 10000):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param level:
    :return:
    """
    for event in event_stream:

      for telescope_id in event.r0.tels_with_data:

            dl1_camera = event.dl1.tel[telescope_id]

            if np.sum(dl1_camera.pe_samples[dl1_camera.cleaning_mask]) >= min_photon:
                yield event


def filter_baseline_zero(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            condition = (np.any(r0_camera.baseline > 0))

            if condition:

                yield event


def filter_trigger_time(event_stream, time):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

            condition = np.sum(output_trigger_patch7) > time

            if condition:

                yield event
