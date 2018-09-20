import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def exponential(x, a, b, c):
    y = a * np.exp(b * x) + c
    return y


class ACLEDInterpolator:

    def __init__(self, ac_level, photo_electrons, photo_electrons_err=None):

        self.ac_level = ac_level
        self.photo_electrons = photo_electrons
        self.photo_electrons_err = photo_electrons_err
        self._interpolate()
        self._extrapolate()
        # self._extrapolate_exponential()

    def __call__(self, ac_level, pixel=None):

        y = self._spline(ac_level)

        y_extrapolated = []

        # for p in self._params:

        #     a = exponential(np.log(ac_level), p[0], p[1], p[2])
        #    y_extrapolated.append(a)
        # y_extrapolated = np.array(y_extrapolated)

        y_extrapolated = np.polyval(self._params.T,
                                    ac_level[:, np.newaxis]).T

        if pixel is not None:

            y = y[pixel]
            y_extrapolated = y_extrapolated[pixel]

        y_shape = y.shape
        y = y.ravel()

        extrapolated_values = np.isnan(y) + (y > 500)
        y_extrapolated = y_extrapolated.ravel()
        y[extrapolated_values] = y_extrapolated[extrapolated_values]
        y = y.reshape(y_shape)

        return y

    def __getitem__(self, item):

        return ACLEDInterpolator(ac_level=self.ac_level,
                                 photo_electrons=self.photo_electrons[item])

    @classmethod
    def load(cls, filename):

        raise NotImplementedError

    def _extrapolate(self):

        pes = self.photo_electrons.copy()
        params = []

        for i, pe in enumerate(pes.T):

            ac_level = self.ac_level.copy()
            mask = (pe > 10) * (pe < 500) * np.isfinite(pe)

            err = None
            if self.photo_electrons_err is not None:
                err = self.photo_electrons_err[:, i]
                mask = mask * (np.isfinite(err))
                err = err[mask]
                err = 1 / err

            pe = pe[mask]
            ac_level = ac_level[mask]

            if len(pe) <= 1:

                param = [np.nan] * 7

                warnings.warn('Could not interpolate pixel {}'.format(i),
                              UserWarning)

            else:
                param = np.polyfit(ac_level, pe, deg=6, w=err)

            params.append(param)

        params = np.array(params)
        self._params = params

    def _extrapolate_exponential(self):

        pes = self.photo_electrons.copy()
        params = []

        for i, pe in enumerate(pes.T):

            print(i)

            ac_level = self.ac_level.copy()
            mask = (pe > 10) * (pe < 500) * np.isfinite(pe)

            err = None
            if self.photo_electrons_err is not None:
                err = self.photo_electrons_err[:, i]
                mask = mask * (np.isfinite(err))
                err = err[mask]
                err = 1 / err

            pe = pe[mask]
            ac_level = ac_level[mask]

            if len(pe) <= 1:

                param = [np.nan] * 3

                warnings.warn('Could not interpolate pixel {}'.format(i),
                              UserWarning)

            else:
                param = curve_fit(exponential, np.log(ac_level),
                                  pe, maxfev=10000)[0]

            params.append(param)

        params = np.array(params)
        self._params = params

    def _interpolate(self):

        pes = self.photo_electrons.copy()
        cubic_spline = interp1d(self.ac_level, pes.T,
                                kind='linear',
                                bounds_error=False,)

        self._spline = cubic_spline

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
                      yerr=y_err, label='Data points, pixel : {}'.format(pixel),
                      linestyle='None', marker='o', color='k', **kwargs)
        axes.plot(x_fit, y_fit, label='Interpolated data', color='r')
        axes.set_xlabel('AC DAC level')
        axes.set_ylabel('Number of p.e.')
        axes.set_yscale('log')
        axes.legend(loc='best')

        return axes


if __name__ == '__main__':

    data = np.load('/home/alispach/ctasoft/digicampipe/charge_linearity_final.npz')
    ac_leds = np.load('/home/alispach/data/tests/mpe/mpe_fit_results.npz')

    ac_levels = ac_leds['ac_levels']
    pe = ac_leds['mu']
    pe_err = ac_leds['mu_error']

    test = ACLEDInterpolator(ac_levels, pe, pe_err)

    test.plot(pixel=559)

    print(test(ac_level=ac_levels, pixel=None).shape)
    plt.show()
