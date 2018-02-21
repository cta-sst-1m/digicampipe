import numpy as np
from digicampipe.utils import calib
from . import Processor


class R1SubtractBaseline(Processor):
    def __call__(event):
        for telescope_id in event.r0.tels_with_data:
            r0 = event.r0.tel[telescope_id]
            r1 = event.r1.tel[telescope_id]

            r1.adc_samples = (
                r0.adc_samples -
                r0.baseline[:, np.newaxis].astype(np.int16)
            )
        return event


class R1FillGainDropAndNsb(Processor):
    def __call__(event):
        for telescope_id in event.r0.tels_with_data:
            r0 = event.r0.tel[telescope_id]
            r1 = event.r1.tel[telescope_id]

            r1.gain_drop = calib.compute_gain_drop(
                r0.standard_deviation, 'std')
            r1.nsb = calib.compute_nsb_rate(r0.standard_deviation, 'std')
        return event


class R1FillGainDropAndNsb_With_DarkBaseline(Processor):
    def __init__(self, dark_baseline):
        self.dark_baseline = dark_baseline

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            r0 = event.r0.tel[telescope_id]
            r1 = event.r1.tel[telescope_id]

            baseline_shift = r0.baseline - self.dark_baseline['baseline']
            r1.gain_drop = calib.compute_gain_drop(baseline_shift, 'mean')
            r1.nsb = calib.compute_nsb_rate(baseline_shift, 'mean')
        return event
