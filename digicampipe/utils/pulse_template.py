import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

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
        return NormalizedPulseTemplate(amplitude=self.amplitude[item],
                                       time=self.time)

    def save(self, filename):
        data = np.vstack([self.time, self.amplitude, self.amplitude_std])
        np.savetxt(filename, data.T)

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
    def create_from_datafile(cls, input_file, min_entries_ratio=0.1):
        """
        Create a template from the 2D histogram file obtained by the
        pulse_shape.py script.
        :param input_file: the 2D histogram file.
        :param min_entries_ratio: minimum ratio of elemnets in a time bin over
        the maximum number of elemnets in any time bin. If a time bin has a
        ratio lower than that, it won't be used in the template.
        :return: the created normalised pulse template object
        """
        histo_pixels = Histogram2d.load(input_file)
        histo = histo_pixels.stack_all(dtype=np.int64)
        tbin_max_entries = np.max(np.sum(histo.contents(), axis=-1))
        # 2 entries minimum per bin to get std values
        min_entries = np.max([2, tbin_max_entries*min_entries_ratio])
        ts, ampl, ampl_std = histo.fit_y(min_entries=min_entries)
        return cls(time=ts[0], amplitude=ampl[0], amplitude_std=ampl_std[0])

    @classmethod
    def create_from_datafiles(cls, input_files, min_entries_ratio=0.1,
                              pixels=None, disable_bar=False):
        """
        Create a template from several 2D histogram files obtained by the
        pulse_shape.py script.
        :param input_files: 2D histogram files.
        :param min_entries_ratio: minimum ratio of elemnets in a time bin over
        the maximum number of elements in any time bin. If a time bin has a
        ratio lower than that, it won't be used in the template.
        :param pixels: list of pixels to take into account.
        :return: the created normalised pulse template object
        """
        sum = None
        for input_file in tqdm(input_files, desc='Files', disable=disable_bar):
            histo_pixels = Histogram2d.load(input_file)
            if sum is None:
                sum = histo_pixels.astype(np.int64)
            else:
                sum += histo_pixels
        histo = sum.stack_all(dtype=np.int64, indexes=pixels)
        tbin_max_entries = np.max(np.sum(histo.contents(), axis=-1))
        # 2 entries minimum per bin to get std values
        min_entries = np.max([2, tbin_max_entries*min_entries_ratio])
        ts, ampl, ampl_std = histo.fit_y(min_entries=min_entries)
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

    def pdf(self, t, t_0):

        y = self(t, t_0=t_0)
        mask = y <= 0
        y[mask] = 1E-10
        integral = np.trapz(y, x=t)
        y = y/ integral

        return y

    def plot(self, axes=None, label='Template data-points', **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        t = np.linspace(self.time.min(), self.time.max(),
                        num=len(self.time) * 100)
        axes.errorbar(self.time, self.amplitude, self.amplitude_std/2,
                      label=label, **kwargs)
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('normalised amplitude [a.u.]')
        axes.legend(loc='best')
        return axes

    def plot_interpolation(self, axes=None, sigma=-1., color='k',
                           label='Template interpolation', cumulative=False,
                           **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        t = np.linspace(self.time.min(), self.time.max(),
                        num=len(self.time) * 100)
        mean_y = self.__call__(t)
        std_y = self.std(t)
        if sigma > 0:
            axes.fill_between(
                t,
                mean_y + sigma * std_y,
                mean_y - sigma * std_y,
                alpha=0.3, color=color, label=label
            )
            axes.plot(t, mean_y, '-', color=color, **kwargs)
        else:
            axes.plot(t, mean_y, '-', label=label, color=color, **kwargs)
        if cumulative:
            integ = np.cumsum(mean_y)
            axes.plot(t, 1 - integ/integ[-1], '--', color=color,
                      label=label + ' integrated', **kwargs)
        axes.legend(loc='best')
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('normalised amplitude [a.u.]')
        return axes