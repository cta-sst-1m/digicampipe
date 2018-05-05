import numpy as np

def fill_electronic_baseline(events):

    for event in events:

        event.data.baseline = event.histo[0].mode

        yield event


def fill_baseline(events, baseline):

    for event in events:

        event.data.baseline = baseline

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
