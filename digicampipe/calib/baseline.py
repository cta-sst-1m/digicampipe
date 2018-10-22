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

        if event.event_type.INTERNAL in event.event_type:
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


def _gain_drop_from_baseline_shift(baseline_shift,
                                   p=np.array([1., -5.252*1e-3,
                                               2.35*1e-5, 5.821*1e-8])):
    """
    Compute the gain drop from baseline shift (assuming nominal gain of
    5.01 LSB/p..e).
    Parameters obtained from Monte-Carlo model of SiPM voltage drop
    :param baseline_shift:
    :param p
    :return:
    """

    return np.polyval(p.T, baseline_shift)


def _crosstalk_drop_from_baseline_shift(baseline_shift,
                                        p=np.array([1., -9.425*1e-3,
                                                    5.463*1e-5, -1.503*1e-8])):
    """
    Compute the crosstalk drop from baseline shift (assuming nominal gain of
    5.01 LSB/p..e).
    Parameters obtained from Monte-Carlo model of SiPM voltage drop
    :param baseline_shift:
    :param p
    :return:
    """

    return np.polyval(p.T, baseline_shift)


def _pde_drop_from_baseline_shift(baseline_shift,
                                  p=np.array([1., -2.187*1e-3, 5.199*1e-6])):
    """
    Compute the PDE (@ 468nm) drop from baseline shift (assuming nominal gain
    of 5.01 LSB/p..e).
    Parameters obtained from Monte-Carlo model of SiPM voltage drop
    :param baseline_shift:
    :param p
    :return:
    """

    return np.polyval(p.T, baseline_shift)


def fill_baseline_r0(event_stream, n_bins=10000):
    n_events = None
    baselines = []
    baselines_std = []
    for event in event_stream:
        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            adc_samples = r0_camera.adc_samples
            if n_events is None:
                n_events = n_bins // adc_samples.shape[1]

            if r0_camera.camera_event_type == 8:
                baselines.append(adc_samples.mean(axis=1))
                baselines = baselines[-n_events:]
                baselines_std.append(adc_samples.std(axis=1))
                baselines_std = baselines_std[-n_events:]

            if len(baselines) == n_events:
                r0_camera.baseline = np.mean(baselines, axis=0)
                r0_camera.standard_deviation = np.mean(baselines_std, axis=0)
        yield event


def compute_baseline_from_waveform(events, bin_left=5, bin_right=10):
    """
    This method will compute the baseline the waveform using the samples
    defined in the region (0, bin_left) U (-bin_right, -1).
    :param events: A stream of events
    :param bin_left: int, Left sample up to which baseline is computed
    :param bin_right: int, Number of samples from the end of the waveform from
    which baseline is computed
    :return:
    """

    for event in events:

        adc_samples = event.data.adc_samples

        adc_samples_first = adc_samples[:, 0:bin_left - 1]
        adc_samples_last = adc_samples[:, -bin_right:]
        adc_samples = np.concatenate((adc_samples_first,
                                      adc_samples_last), axis=1)

        baseline = np.mean(adc_samples, axis=-1)
        std = np.std(adc_samples, axis=-1)

        event.data.baseline = baseline
        event.data.baseline_std = std

        yield event


def compute_baseline_simtel(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            n_pixels = event.inst.num_pixels[telescope_id]
            r0_camera = event.r0.tel[telescope_id]
            r0_camera.baseline = (event.mc.tel[telescope_id].pedestal[0]
                                  / event.r0.tel[telescope_id].num_samples
                                  )
            # standard_deviation should be set up manualy and it should be
            # probably equal to value of 'fadc_noise' variable from simtel
            # configuration file
            r0_camera.standard_deviation = 1.5 * np.ones(n_pixels)

        yield event
