import numpy as np

__all__ = ['tag_burst_from_moving_average_baseline']


def tag_burst_from_moving_average_baseline(events, n_previous_events=100,
                                           threshold_lsb=5):
    last_mean_baselines = []
    last_time = None
    for event in events:
        mean_baseline = np.mean(event.data.baseline)
        time = event.data.local_time
        # reset buffer if there is a gap > 30s
        if last_time is not None and time - last_time > 30:
            last_mean_baselines = []
        if len(last_mean_baselines) != n_previous_events:
            last_mean_baselines.append(mean_baseline)
        else:
            last_mean_baselines = last_mean_baselines[1:]
            last_mean_baselines.append(mean_baseline)
        moving_avg_baseline = np.mean(last_mean_baselines)
        if (mean_baseline - moving_avg_baseline) > threshold_lsb:
            event.data.burst = True
        else:
            event.data.burst = False
        yield event


def tag_border_events(events, geom, skip=False):
    for event in events:
        mask = event.data.cleaning_mask
        num_neighbors = np.sum(geom.neighbor_matrix[mask], axis=-1)
        event.data.border = np.any(num_neighbors < 6)
        if event.data.border and skip:
            continue
        yield event
