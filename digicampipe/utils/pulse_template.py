import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class NormalizedPulseTemplate:
    def __init__(self, amplitude, time):

        dt = np.diff(time)

        if not (dt == dt[0]).all():
            raise ValueError('time argument should be of constant sampling')

        self.time = time
        self.amplitude = amplitude
        self._template = self._interpolate()

    def __call__(self, time, amplitude=1, t_0=0, baseline=0):

        y = amplitude * self._template(time - t_0) + baseline

        return np.array(y)

    def __getitem__(self, item):

        return NormalizedPulseTemplate(amplitude=self.amplitude[item],
                                       time=self.time)

    @classmethod
    def load(cls, filename):

        t, x = np.loadtxt(filename).T

        return cls(amplitude=x, time=t)

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
