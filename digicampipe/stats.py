import numpy as np


def stats_events(event_stream):

    times_above_threshold = []
    patches_above_threshold = []
    trigger_times = []


    for event_number, event in enumerate(event_stream):

        print(event_number)

        for telescope_id in event.r0.tels_with_data:

            output_trigger_patch7 = np.array(list(event.r0.tel[telescope_id].trigger_output_patch7.values()))
            time_above_threshold = np.sum(output_trigger_patch7, axis=1)
            patch_above_threshold = np.sum((time_above_threshold > 0))
            trigger_time = event.r0.tel[telescope_id].local_camera_clock

            patches_above_threshold.append(patch_above_threshold)
            time_above_threshold = np.sum(time_above_threshold, axis=0)
            times_above_threshold.append(time_above_threshold)
            trigger_times.append(trigger_time)

    return np.array(patches_above_threshold), np.array(times_above_threshold), np.array(trigger_times)
