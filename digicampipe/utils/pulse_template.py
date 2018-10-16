import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.baseline import fill_digicam_baseline, \
    subtract_baseline, correct_wrong_baseline
from digicampipe.calib.time import estimate_time_from_leading_edge
from digicampipe.utils.hist2d import Histogram2dChunked


class NormalizedPulseTemplate:
    def __init__(self, amplitude, time, amplitude_std=None):
        self.time = time
        self.amplitude = amplitude
        if amplitude_std is not None:
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
            amplitude_range=(-.1, 0.4), n_bin=100
    ):
        events = calibration_event_stream(input_files)
        events = fill_digicam_baseline(events)
        if "SST1M_01_201805" in input_files[0]:  # fix data in May
            print("WARNING: correction of the baselines applied.")
            events = correct_wrong_baseline(events)
        events = subtract_baseline(events)
        histo = None
        n_sample = 0
        n_pixel = 0
        for e in events:
            adc = e.data.adc_samples
            integral = adc[:, 10:30].sum(axis=1)
            adc_norm = adc[pixels, :] / integral[pixels, None]
            arrival_time_in_ns = estimate_time_from_leading_edge(adc) * 4
            if histo is None:
                n_pixel, n_sample = adc_norm.shape
                histo = Histogram2dChunked(
                    shape=(n_pixel, n_bin, n_bin),
                    range=[time_range_ns, amplitude_range]
                )
            else:
                assert adc_norm.shape[0] == n_pixel
                assert adc_norm.shape[1] == n_sample
            time_in_ns = np.arange(n_sample) * 4
            # charge < 10 pe (noisy) or > 500 pe (saturation) => bad_charge
            # 1 pe <=> 20 integral
            bad_charge = np.logical_or(integral < 200, integral > 10000)
            arrival_time_in_ns[bad_charge] = -np.inf # ignored by histo
            histo.fill(
                x=time_in_ns[None, :] - arrival_time_in_ns[pixels, None],
                y=adc_norm
            )
        t_pixels, ampl_pixels, ampl_std_pixels = histo.fit_y()
        if n_pixel == 1:
            return cls(amplitude=ampl_pixels[0], time=t_pixels[0],
                       amplitude_std=ampl_std_pixels[0])
        ts = np.unique(np.concatenate(t_pixels))
        if len(ts) < 2:
            raise RuntimeError('no charge passed the cuts')
            return
        ampl = np.zeros_like(ts)
        ampl_std = np.zeros_like(ts)
        for idx, t in enumerate(ts):
            ampl_sum_t = 0
            n_pixel_t = 0
            for pixel in range(n_pixel):
                bool_pos = t == t_pixels[pixel]
                if np.any(bool_pos):
                    n_pixel_t += 1
                    ampl_sum_t += ampl_pixels[pixel][bool_pos]
            ampl[idx] = ampl_sum_t / n_pixel_t
            ampl_var1_t = 0
            ampl_var2_t = 0
            for pixel in range(n_pixel):
                bool_pos = t == t_pixels[pixel]
                if np.any(bool_pos):
                    ampl_var1_t += ampl_std_pixels[pixel][bool_pos] ** 2
                    ampl_var2_t += (ampl_pixels[pixel][bool_pos] - ampl[idx]) ** 2
            if n_pixel_t > 1:
                ampl_std[idx] = np.sqrt((ampl_var1_t + ampl_var2_t)/(n_pixel_t - 1))
            elif n_pixel_t == 1:
                ampl_std[idx] = np.sqrt(ampl_var1_t)
            else:
                raise RuntimeError('unexpected problem calulating std')
        return cls(time=ts, amplitude=ampl, amplitude_std=ampl_std)

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
