import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.baseline import fill_digicam_baseline, subtract_baseline
from digicampipe.calib.time import estimate_time_from_leading_edge
from digicampipe.utils.hist2d import Histogram2dChunked


class NormalizedPulseTemplate:
    def __init__(self, amplitude, time, amplitude_std=None):
        dt = np.diff(time)
        if not (dt == dt[0]).all():
            raise ValueError('time argument should be of constant sampling')
        self.time = time
        self.amplitude = amplitude
        if amplitude_std:
            assert len(amplitude_std) == len(amplitude)
            self.amplitude_std = amplitude_std
        else:
            self.amplitude_std = self.amplitude * 0
        self._template = self._interpolate()

    def __call__(self, time, amplitude=1, t_0=0, baseline=0):
        y = amplitude * self._template(time - t_0) + baseline
        return np.array(y)

    def __getitem__(self, item):
        print(self.amplitude[item], item)
        return NormalizedPulseTemplate(amplitude=self.amplitude[item],
                                       time=self.time)

    @classmethod
    def load(cls, filename):
        data = np.loadtxt(filename).T
        assert len(data) in [2, 3]
        if len(data) == 2:  # no std in file
            t, x = data
            return cls(amplitude=x, time=t)
        elif len(data) == 3:
            t, x, dx = data
            return cls(amplitude=x, time=t, amplitude_std=dx)

    @classmethod
    def create_from_datafiles(
            cls, input_files, pixels=(0,), time_range_ns=(-10, 40),
            amplitude_range=(-.1, 0.4), n_bin=101
    ):
        events = calibration_event_stream(input_files)
        events = fill_digicam_baseline(events)
        events = subtract_baseline(events)
        histo = None
        n_sample = 0
        n_pixel = 0
        for e in events:
            adc = e.data.adc_samples
            integral = adc[:, 10:30].sum(axis=1)
            adc_norm = adc / integral[:, None]
            assert np.all(adc_norm[:, 10:30].sum(axis=1) == 1)
            arrival_time_in_ns = estimate_time_from_leading_edge(adc) * 4
            if histo is None:
                n_pixel, n_sample = adc[pixels, :].shape
                histo = Histogram2dChunked(
                    shape=(n_pixel, n_bin, n_bin),
                    range=[time_range_ns, amplitude_range]
                )
            else:
                assert adc.shape == n_pixel, n_sample
            time_in_ns = np.arange(n_sample) * 4
            histo.fill(
                x=time_in_ns[None, :] - arrival_time_in_ns[pixels, None],
                y=adc_norm[pixels, :]
            )
        t, x, dx = histo.fit_y()
        return cls(amplitude=x, time=t, amplitude_std=dx)

    def _interpolate(self):
        if abs(np.min(self.amplitude)) <= abs(np.max(self.amplitude)):

            normalization = np.max(self.amplitude)

        else:

            normalization = np.min(self.amplitude)

        self.amplitude = self.amplitude / normalization

        return interp1d(self.time, self.amplitude, kind='cubic',
                        bounds_error=False, fill_value=0., assume_sorted=True)

    def integral(self):

        return np.trapz(y=self.amplitude, x=self.time)

    def compute_charge_amplitude_ratio(self, integral_width, dt_sampling):

        dt = self.time[1] - self.time[0]

        if not dt % dt_sampling:
            raise ValueError('Cannot use sampling rate {} for {}'.format(
                1 / dt_sampling, dt))
        step = int(dt_sampling / dt)
        y = self.amplitude[::step]
        window = np.ones(integral_width)
        charge_to_amplitude_factor = np.convolve(y, window)
        charge_to_amplitude_factor = np.max(charge_to_amplitude_factor)

        return 1 / charge_to_amplitude_factor

    def plot(self, axes=None, **kwargs):

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        t = np.linspace(self.time.min(), self.time.max(),
                        num=len(self.time) * 100)

        axes.plot(self.time, self.amplitude, label='Template data-points',
                  **kwargs)
        axes.plot(t, self(t), label='Interpolated template')
        axes.legend(loc='best')

        return axes
