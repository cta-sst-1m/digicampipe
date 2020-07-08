import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod
import fitsio


def exponential(x, a, b):

    y = a * np.exp(b * x)

    return y


class LightSource(ABC):

    def __init__(self, x, y, y_err=None):

        y = np.atleast_2d(y)

        assert x[0] == 0
        assert (np.diff(x) > 0).all()

        if y_err is not None:

            y_err = np.atleast_2d(y_err)
            assert y.shape == y_err.shape

        self.x = x
        self.y = y
        self.y_err = y_err
        self._interpolator = self._interpolate()

    def __call__(self, x, pixel=None, *args, **kwargs):

        y_interpolated = self._interpolator(x, *args, **kwargs)

        if pixel is not None:

            y_interpolated = y_interpolated[pixel]

        return y_interpolated

    def __getitem__(self, item):

        return self.__class__(x=self.x, y=self.y[item],
                              y_err=self.y_err[item])

    @abstractmethod
    def _interpolate(self):

        pass

    @classmethod
    @abstractmethod
    def load(cls, filename):

        pass

    def save(self, filename):

        with fitsio.FITS(filename, 'rw', clobber=True) as f:

            f.write(self.x, extname='x')
            f.write(self.y, extname='y')
            f.write(self.y_err, extname='y_err')


class ACLED(LightSource):

    saturation_threshold = 300  # in p.e.

    def __init__(self, x, y, y_err=None):

        self.cubic_spline = None
        self.params_polynomial = None
        self.params_exponential = None

        super().__init__(x, y, y_err)

    @classmethod
    def load(cls, filename):

        with fitsio.FITS(filename, 'r') as f:

            x = f['x'].read()
            y = f['y'].read()
            y_err = f['y_err'].read()

        obj = ACLED(x, y, y_err)

        return obj

    def _interpolate(self):

        self._fit_spline()
        self._fit_polynomial()
        self._fit_exponential()

        def _func(x):

            y = self.func_exponential(x)
            y_shape = y.shape
            y = y.ravel()
            y_spline = self.func_spline(x).ravel()
            y_poly = self.func_polynomial(x).ravel()

            start_values = (y < 5)
            end_values = (y > self.saturation_threshold)

            y[~start_values] = y_spline[~start_values]
            y[end_values] = y_poly[end_values]
            y = y.reshape(y_shape)

            return y

        return _func

    def _fit_spline(self):

        pes = self.y.copy()
        cubic_spline = []

        for pe in pes:

            mask = np.isfinite(pe)  # * (pe > 5)
            x = self.x[mask]
            y = pe[mask]

            if not len(x):
                x = self.x
                y = pe

            spline = interp1d(x, y,
                              kind='quadratic',
                              bounds_error=False,
                              fill_value=np.nan)

            cubic_spline.append(spline)

        self.cubic_spline = cubic_spline

    def _fit_polynomial(self):

        pes = self.y.copy()
        params = []
        deg = 4

        for i, pe in enumerate(pes):

            ac_level = self.x.copy()
            mask = (pe > 50) * (pe < self.saturation_threshold) * np.isfinite(pe)

            err = None
            if self.y_err is not None:
                err = self.y_err[i]
                mask = mask * (np.isfinite(err))
                err = err[mask]
                err = 1 / err

            pe = pe[mask]
            ac_level = ac_level[mask]

            err = None

            if len(pe) <= 1:

                param = [np.nan] * (deg + 1)

                warnings.warn('Could not interpolate pixel {}'.format(i),
                              UserWarning)

            else:
                param = np.polyfit(ac_level, pe, deg=deg, w=err)

            params.append(param)

        params = np.array(params)
        self.params_polynomial = params

    def _fit_exponential(self):

        pes = self.y.copy()
        params = []

        for i, pe in enumerate(pes):

            ac_level = self.x.copy()
            mask = (pe > 0) * (pe < 100) * np.isfinite(pe) * (ac_level >= 0)

            if self.y_err is not None:
                err = self.y_err[i]
                mask = mask * (np.isfinite(err))
                err = err[mask]

            else:

                err = None

            pe = pe[mask]
            ac_level = ac_level[mask]

            if len(pe) <= 5:

                param = [np.nan] * 2

                warnings.warn('Could not interpolate pixel {}'.format(i),
                              UserWarning)

            else:

                try:

                    intercept = np.where(ac_level == np.min(ac_level))[0][0]
                    intercept = pe[intercept]

                    slope = np.log(pe)
                    slope = np.diff(slope) / np.diff(ac_level)
                    slope = np.mean(slope)

                    p0 = np.array([intercept, slope])

                    param = curve_fit(exponential, ac_level,
                                              pe, maxfev=10000, sigma=err,
                                      p0=p0)[0]

                except RuntimeError:

                    param = [np.nan] * 2

                    warnings.warn('Could not interpolate pixel {}'.format(i),
                                  UserWarning)

            params.append(param)

        params = np.array(params)
        self.params_exponential = params

    def func_exponential(self, x):

        y = np.zeros((len(self.y), len(x)))

        for i in range(len(y)):
            y[i] = exponential(x, self.params_exponential[i][0],
                               self.params_exponential[i][1])

        return y

    def func_spline(self, x):

        y = np.zeros((len(self.y), len(x)))

        for i in range(len(y)):
            y[i] = self.cubic_spline[i](x)

        return y

    def func_polynomial(self, x):

        y = np.zeros((len(self.y), len(x)))

        for i in range(len(y)):
            y[i] = np.polyval(self.params_polynomial[i], x).T

        return y

    def plot(self, axes=None, pixel=0, y_lim=(0, 2000), show_fits=True,
             **kwargs):

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        x_fit = np.arange(1000)
        y_fit = self(x_fit, pixel=pixel)

        mask = (y_fit > y_lim[0]) * (y_fit <= y_lim[1])
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]

        if self.y_err is not None:

            y_err = self.y_err[pixel]
        else:

            y_err = None

        axes.errorbar(self.x, self.y[pixel],
                      yerr=y_err, label='Data points, pixel : {}'.format(pixel),
                      linestyle='None', marker='o', color='k', **kwargs)
        axes.plot(x_fit, y_fit, label='Interpolated data', color='r')

        if show_fits:
            axes.plot(x_fit, self.func_spline(x_fit)[pixel], label='Spline')
            axes.plot(x_fit, self.func_polynomial(x_fit)[pixel], label='Polynomial')
            axes.plot(x_fit, self.func_exponential(x_fit)[pixel], label='Exponential')
        axes.set_xlabel('AC DAC level')
        axes.set_ylabel('Number of p.e.')
        axes.set_yscale('log')
        axes.legend(loc='best')

        return axes


class DCLED(LightSource):

    def __init__(self, x, y, y_err):

        super().__init__(x, y, y_err)

    @classmethod
    def load(cls, filename):

        with fitsio.FITS(filename, 'r') as f:

            x = f['x'].read()
            y = f['y'].read()
            y_err = f['y_err'].read()

        obj = DCLED(x, y, y_err)

        return obj

    def _interpolate(self):

        pass

    def plot(self):

        pass
