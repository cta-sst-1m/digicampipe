import numpy as np

from digicampipe.utils import Processor


class FillBaseline(Processor):
    ''' calculate baseline from previous events and store it in the event.
    '''
    def __init__(self, n_bins=10000):
        self.n_bins = n_bins
        self.count_calib_events = 0

        self.means = []
        self.stds = []

        self.never_seen_an_event = True

    def _init(self, event, telescope_id):
        self.never_seen_an_event = False
        self.n_pixels = event.inst.num_pixels[telescope_id]
        self.n_samples = event.inst.num_samples[telescope_id]
        self.n_events = self.n_bins // self.n_samples
        self.baselines = np.zeros((self.n_pixels, self.n_events))
        self.baselines_std = np.zeros((self.n_pixels, self.n_events))
        self.baseline = np.zeros(self.n_pixels)
        self.std = np.zeros(self.n_pixels)

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:

            if self.never_seen_an_event:
                self._init(event, telescope_id)

            r0_camera = event.r0.tel[telescope_id]

            if r0_camera.camera_event_type == 8:
                self.count_calib_events += 1

                adc_samples = r0_camera.adc_samples
                new_mean = np.mean(adc_samples, axis=-1)
                new_std = np.std(adc_samples, axis=-1)

                self.baselines = np.roll(self.baselines, 1, axis=-1)
                self.baselines_std = np.roll(self.baselines_std, 1, axis=-1)

                self.baseline += new_mean - self.baselines[..., 0]
                self.std += new_std - self.baselines_std[..., 0]

                self.baselines[..., 0] = new_mean
                self.baselines_std[..., 0] = new_std

            if self.count_calib_events >= self.n_events:

                r0_camera.baseline = self.baseline / self.n_events
                r0_camera.standard_deviation = self.std / self.n_events

            else:

                r0_camera.baseline = np.zeros(self.n_pixels) * np.nan
                r0_camera.standard_deviation = np.zeros(self.n_pixels) * np.nan
        return event
