import numpy as np


def save_external_triggers(
    data_stream,
    output_filename,
    n_events=None,
    pixel_list=[...]
):

    description = {
        'baseline': lambda e, t, pl: list(e.r0.tel[t].baseline[pl]),
        'baseline_dark': lambda e, t, pl: list(e.r0.tel[t].dark_baseline[pl]),
        'baseline_shift': lambda e, t, pl:
            list(e.r0.tel[t].baseline[pl] - e.r0.tel[t].dark_baseline[pl]),
        'baseline_std':
            lambda e, t, pl: list(e.r0.tel[t].standard_deviation[pl]),
        'nsb_rate': lambda e, t, pl: list(e.r1.tel[t].nsb[pl]),
        'gain_drop': lambda e, t, pl: list(e.r1.tel[t].gain_drop[pl]),
        'time_stamp': lambda e, t, pl: e.r0.tel[t].local_camera_clock,
    }

    output = {name: [] for name in description}

    for i, event in enumerate(data_stream):
        if i >= n_events:
            break

        for telescope_id in event.r0.tels_with_data:
            for name, getter in description.items():
                output[name].append(getter(event, telescope_id, pixel_list))

    output = {name: np.array(value) for name, value in output.items()}
    np.savez(output_filename, **output)
