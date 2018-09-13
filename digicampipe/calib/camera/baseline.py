import numpy as np

__all__ = ['fill_dark_baseline', 'fill_baseline', 'fill_digicam_baseline',
           'compute_baseline_with_min', 'subtract_baseline',
           'compute_baseline_shift', 'compute_baseline_std',
           'compute_nsb_rate', 'compute_gain_drop']


def fill_dark_baseline(events, dark_baseline):

    for event in events:

        event.data.dark_baseline = dark_baseline

        yield event


def fill_baseline(events, baseline):

    for event in events:

        event.data.baseline = baseline

        yield event


def fill_digicam_baseline(events):

    for event in events:

        event.data.baseline = event.data.digicam_baseline

        yield event


def compute_baseline_with_min(events):

    for event in events:

        adc_samples = event.data.adc_samples
        event.data.baseline = np.min(adc_samples, axis=-1)

        yield event


def subtract_baseline(events):

    for event in events:

        baseline = event.data.baseline

        event.data.adc_samples = event.data.adc_samples.astype(baseline.dtype)
        event.data.adc_samples -= baseline[..., np.newaxis]

        yield event


def compute_baseline_shift(events):

    for event in events:

        event.data.baseline_shift = event.data.baseline \
                                    - event.data.dark_baseline

        yield event


def compute_baseline_std(events, n_events):

    baselines_std = []
    for event in events:

        data = event.data.adc_samples

        if event.event_type == 8:

            baselines_std.append(data.std(axis=1))
            baselines_std = baselines_std[-n_events:]
            event.data.baseline_std = np.mean(baselines_std, axis=0)

        if len(baselines_std) == n_events:

            yield event


def compute_nsb_rate(events, gain, pulse_area, crosstalk, bias_resistance,
                     cell_capacitance):

    for event in events:

        baseline_shift = event.data.baseline_shift
        nsb_rate = baseline_shift / (gain * pulse_area * (1 + crosstalk) -
                                     baseline_shift * bias_resistance *
                                     cell_capacitance)
        event.data.nsb_rate = nsb_rate

        yield event


def compute_gain_drop(events, bias_resistance, cell_capacitance):

    for event in events:

        nsb_rate = event.data.nsb_rate
        gain_drop = 1. / (1. + nsb_rate * cell_capacitance
                          * bias_resistance)
        gain_drop = gain_drop.value
        event.data.gain_drop = gain_drop

        yield event


def tag_burst(events, event_average=100, threshold_lsb=2):
    last_mean_baselines = []
    for event in events:
        mean_baseline = np.mean(event.data.digicam_baseline)
        if len(last_mean_baselines) != event_average:
            last_mean_baselines.append(mean_baseline)
        else:
            last_mean_baselines = last_mean_baselines[1:]
            last_mean_baselines.append(mean_baseline)
        moving_avg_baseline = np.mean(last_mean_baselines)
        event.data.baseline_running_average = moving_avg_baseline
        if (mean_baseline - moving_avg_baseline) > threshold_lsb:
            event.data.burst = True
        else:
            event.data.burst = False
        yield event
