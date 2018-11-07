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

    def save(self, filename):

        with fitsio.FITS(filename, 'rw', clobber=True) as f:

            f.write(self.x, extname='x')
            f.write(self.y, extname='y', compress='gzip')
            f.write(self.y_err, extname='y_err', compress='gzip')

    def plot(self, axes=None, pixel=0, **kwargs):

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        x_fit = np.arange(1000)
        y_fit = self(x_fit, pixel=pixel)

        mask = (y_fit > 0) * (y_fit <= 2000)
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]

        if self.photo_electrons_err is not None:

            y_err = self.photo_electrons_err[:, pixel]
        else:

            y_err = None

        axes.errorbar(self.ac_level, self.photo_electrons[:, pixel],
                      yerr=y_err,
                      label='Data points, pixel : {}'.format(pixel),
                      linestyle='None', marker='o', color='k', **kwargs)
        axes.plot(x_fit, y_fit, label='Interpolated data', color='r')
        axes.set_xlabel('AC DAC level')
        axes.set_ylabel('Number of p.e.')
        axes.set_yscale('log')
        axes.legend(loc='best')

        return axes


class ACLED(LightSource):

    saturation_threshold = 300  # in p.e.

    def __init__(self, ac_level, photo_electrons, photo_electrons_err=None):

        assert ac_level[0] == 0
        assert (np.diff(ac_level) > 0).all()

        self.x = ac_level - ac_level[0]
        self.y = photo_electrons
        self.y_err = photo_electrons_err
        self._interpolate()
        self._extrapolate()
        self._extrapolate_exponential()

    @classmethod
    def load(cls, filename):

        with fitsio.FITS(filename, 'r') as f:
            x = f['x'].read()
            y = f['y'].read()
            y_err = f['y_err'].read()

        obj = ACLED(x, y, y_err)

        return obj

    def __call__(self, ac_level, pixel=None):

        y = self.func_exponential(ac_level)
        y_shape = y.shape
        y = y.ravel()
        y_spline = self.func_spline(ac_level).ravel()
        y_poly = self.func_polynomial(ac_level).ravel()

        start_values = (y < 5)
        end_values = (y > self.saturation_threshold)

        y[~start_values] = y_spline[~start_values]
        y[end_values] = y_poly[end_values]
        y = y.reshape(y_shape)

        if pixel is not None:

            y = y[pixel]

        return y

    def _extrapolate(self):

        pes = self.y.copy()
        params = []
        deg = 4

        for i, pe in enumerate(pes.T):

            ac_level = self.x.copy()
            mask = (pe > 50) * (pe < self.saturation_threshold) * np.isfinite(pe)

            err = None
            if self.y_err is not None:
                err = self.y_err[:, i]
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
        self._params = params

    def _extrapolate_exponential(self):

        pes = self.y.copy()
        params = []

        for i, pe in enumerate(pes.T):

            ac_level = self.x.copy()
            mask = (pe > 0) * (pe < 100) * np.isfinite(pe) * (ac_level >= 0)

            if self.y_err is not None:
                err = self.y_err[:, i]
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
        self._params_exponential = params

    def func_exponential(self, x):

        y = np.zeros((self.y.shape[1], len(x)))

        for i in range(len(y)):
            y[i] = exponential(x, self._params_exponential[i][0],
                               self._params_exponential[i][1])

        return y

    def func_spline(self, x):

        y = np.zeros((self.y.shape[1], len(x)))

        for i in range(len(y)):
            y[i] = self._spline[i](x)

        return y

    def func_polynomial(self, x):

        y = np.zeros((self.y.shape[1], len(x)))

        for i in range(len(y)):
            y[i] = np.polyval(self._params[i], x).T

        return y

    def _interpolate(self):

        pes = self.y.copy().T
        cubic_spline = []

        for pe in pes:

            mask = np.isfinite(pe) # * (pe > 5)
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

        # mask = np.isfinite(pes)
        # pes[~mask] = 0
        # w = np.ones(pes.shape)
        # w[~mask] = 0
        # w = np.sum(w, axis=0)
        # w[w>0] = 1
        #
        # print(pes.shape, w.shape, self.ac_level.shape)
        # cubic_spline = splprep(pes.T, w=w.T, u=self.ac_level)
        #
        # # cubic_spline = interp1d(self.ac_level, pes.T,
         #                       kind='slinear',
         #                       bounds_error=None,
         #                       fill_value='extrapolate')

        self._spline = cubic_spline

    def plot(self, axes=None, pixel=0, y_lim=(0, 2000), show_fits=True, **kwargs):

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        x_fit = np.arange(1000)
        y_fit = self(x_fit, pixel=pixel)

        mask = (y_fit > y_lim[0]) * (y_fit <= y_lim[1])
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]

        if self.y_err is not None:

            y_err = self.y_err[:, pixel]
        else:

            y_err = None

        axes.errorbar(self.x, self.y[:, pixel],
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

    def __init__(self, dc_level, nsb_rate, nsb_rate_error=None):

        self.dc_level = dc_level
        self.nsb_rate = nsb_rate
        self.nsb_rate_error = nsb_rate_error
        self._interpolate()

    def __call__(self, *args, **kwargs):

        pass

    def __getitem__(self, item):

        return DCLED(dc_level=self.dc_level, nsb_rate=self.nsb_rate[item],
                     nsb_rate_error=self.nsb_rate_error[item])

    def load(cls, filename):

        pass

    def save(self, filename):

        pass

    def plot(self):

        plt.figure()


if __name__ == '__main__':

    # data = np.load('/home/alispach/Documents/PhD/ctasoft/digicampipe/charge_linearity_final.npz')
    ac_leds = np.load('/sst1m/analyzed/calib/mpe/mpe_fit_results_combined.npz')

    ac_levels = ac_leds['ac_levels'][:, 0]

    pe = ac_leds['mu']


    pe_err = ac_leds['mu_error']
    print(ac_levels.tolist())
    print(pe[:, 0:2].tolist())
    print(pe_err[:, 0:2].tolist())


    test = ACLED(ac_levels, pe, pe_err)

    x = np.linspace(0, 1000, num=1E3)
    # X = np.zeros((1296, len(x)))
    # X[:] = x
    print(test(x).shape)

    test.plot(pixel=1, y_lim=(0, 1E4), show_fits=False)

    y = test(x)

    X = np.zeros(y.shape)
    X[:] = x

    mask = np.isfinite(X) * np.isfinite(y)

    y = y[mask].ravel()
    x = X[mask].ravel()

    print(x.shape, y.shape)

    x_min, x_max = max(1, np.min(x)), np.max(x)
    y_min, y_max = max(1, np.min(y)), np.max(y)

    n_bins_x, n_bins_y = 150, 150

    # bin_x = np.logspace(np.log10(x_min), np.log10(x_max), num=n_bins_x)
    bin_x = np.linspace(x_min, x_max, num=n_bins_x)
    bin_y = np.logspace(np.log10(y_min), np.log10(y_max), num=n_bins_y)

    plt.figure()
    plt.hist2d(x, y, bins=[bin_x, bin_y], range=[[x_min, x_max], [y_min, y_max]]
               )#, normed=True)
    plt.yscale('log')
    plt.xlabel('AC DAC level')
    plt.ylabel('Number of p.e.')
    plt.colorbar(label='Counts []')

    plt.show()

