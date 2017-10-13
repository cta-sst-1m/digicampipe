import numpy as np
import itertools


def save_external_triggers(data_stream, output_filename, n_events=None, pixel_list=[...]):

    baseline = []
    baseline_dark = []
    baseline_shift = []
    baseline_std = []
    nsb_rate = []
    gain_drop = []
    time_stamp = []

    if n_events is None:

        iter_events = itertools.count()

    else:

        iter_events = range(n_events)

    for event, i in zip(data_stream, iter_events):

        for telescope_id in event.r0.tels_with_data:

            baseline.append(list(event.r0.tel[telescope_id].baseline[pixel_list]))
            baseline_dark.append(list(event.r0.tel[telescope_id].dark_baseline[pixel_list]))
            baseline_shift.append(list(event.r0.tel[telescope_id].baseline[pixel_list] - event.r0.tel[telescope_id].dark_baseline[pixel_list]))
            baseline_std.append(list(event.r0.tel[telescope_id].standard_deviation[pixel_list]))
            nsb_rate.append(list(event.r1.tel[telescope_id].nsb[pixel_list]))
            gain_drop.append(list(event.r1.tel[telescope_id].gain_drop[pixel_list]))
            time_stamp.append(event.r0.tel[telescope_id].local_camera_clock)

    np.savez(output_filename,
             baseline=np.array(baseline),
             baseline_dark=np.array(baseline_dark),
             baseline_shift=np.array(baseline_shift),
             baseline_std=np.array(baseline_std),
             nsb_rate=np.array(nsb_rate),
             gain_drop=np.array(gain_drop),
             time_stamp=np.array(time_stamp))


