import numpy as np


def save_mean_event_baseline(event_stream, filename):
    baseline = []
    std = []

    for i, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:
            dl1_camera = event.dl1.tel[telescope_id]
            r0_camera = event.r0.tel[telescope_id]

            baseline.append(np.mean(
                r0_camera.baseline[dl1_camera.cleaning_mask]))
            std.append(np.mean(
                r0_camera.standard_deviation[dl1_camera.cleaning_mask]))

        yield event

    np.savetxt(filename, np.column_stack((baseline, std)), '%1.5f')
