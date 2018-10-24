import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from digicampipe.utils.hist2d import Histogram2d


class NormalizedPulseTemplate:
    def __init__(self, amplitude, time, amplitude_std=None):
        self.time = np.array(time)
        self.amplitude = np.array(amplitude)
        if amplitude_std is not None:
            assert np.array(amplitude_std).shape == self.amplitude.shape
            self.amplitude_std = np.array(amplitude_std)
        else:
            self.amplitude_std = self.amplitude * 0
        self._template = self._interpolate()
        self._template_std = self._interpolate_std()

    def __call__(self, time, amplitude=1, t_0=0, baseline=0):
        y = amplitude * self._template(time - t_0) + baseline
        return np.array(y)

    def std(self, time, amplitude=1, t_0=0, baseline=0):
        y = amplitude * self._template_std(time - t_0) + baseline
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
    def create_from_datafile(cls, input_file):
        """
        Create a template from the 2D histogram file obtained by the
        pulse_shape.py script.
        """
        histo_pixels = Histogram2d.load(input_file)
        histo = histo_pixels.stack_all(dtype=np.int64)
        ts, ampl, ampl_std = histo.fit_y(min_entries=10000)
        return cls(time=ts[0], amplitude=ampl[0], amplitude_std=ampl_std[0])

    def _interpolate(self):
        if abs(np.min(self.amplitude)) <= abs(np.max(self.amplitude)):

            normalization = np.max(self.amplitude)

        else:

            normalization = np.min(self.amplitude)

        self.amplitude = self.amplitude / normalization
        self.amplitude_std = self.amplitude_std / normalization

        return interp1d(self.time, self.amplitude, kind='cubic',
                        bounds_error=False, fill_value=0., assume_sorted=True)

    def _interpolate_std(self):
        return interp1d(self.time, self.amplitude_std, kind='cubic',
                        bounds_error=False, fill_value=np.inf,
                        assume_sorted=True)

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

        axes.errorbar(self.time, self.amplitude, self.amplitude_std,
                      label='Template data-points', **kwargs)
        axes.plot(t, self(t), '-', label='Interpolated template')
        axes.legend(loc='best')
        return axes
