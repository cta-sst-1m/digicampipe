import numpy as np


def filter_patch(event_stream, unwanted_patch, reversed=False):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

            patch_condition = np.any(output_trigger_patch7[unwanted_patch])
            if (not reversed) and patch_condition:

                yield event

            elif reversed and (not patch_condition):

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
