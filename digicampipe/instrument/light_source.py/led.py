import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import lambertw


def exponential(x, a, b):

    # log_y = np.log(a) + b * x
    # y = np.exp(log_y)
    y = a * np.exp(b * x)

    return y


class ACLEDInterpolator:

    def __init__(self, ac_level, photo_electrons, photo_electrons_err=None):

        self.ac_level = ac_level
        self.photo_electrons = photo_electrons
        self.photo_electrons_err = photo_electrons_err
        self._interpolate()
        # self._extrapolate()
        self._extrapolate_exponential()

    def __call__(self, ac_level, pixel=None):

        y = np.zeros((self.photo_electrons.shape[1], len(ac_level)))
        y_extrapolated = np.zeros((self.photo_electrons.shape[1], len(ac_level)))

        for i in range(len(y)):

            y[i] = self._spline[i](ac_level)
            y_extrapolated[i] = exponential(ac_level,
                                            self._params[i][0],
                                            self._params[i][1])
        # y = self._spline(ac_level)
        # y = splev(ac_level, self._spline[0])

        # y_extrapolated = np.polyval(self._params.T,
        # ac_level[:, np.newaxis]**5).T

        if pixel is not None:

            y = y[pixel]
            y_extrapolated = y_extrapolated[pixel]

        y_shape = y.shape
        y = y.ravel()

        extrapolated_values = np.isnan(y) + (y > 500)
        y_extrapolated = y_extrapolated.ravel()
        y[extrapolated_values] = y_extrapolated[extrapolated_values]
        y = y.reshape(y_shape)

        # print(y.shape)
        # print(self._spline.fill_value)

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

            err = None

            if len(pe) <= 1:

                param = [np.nan] * 2

                warnings.warn('Could not interpolate pixel {}'.format(i),
                              UserWarning)

            else:
                param = np.polyfit(ac_level**5, pe, deg=1, w=err)

            params.append(param)

        params = np.array(params)
        self._params = params

    def _extrapolate_exponential(self):

        pes = self.photo_electrons.copy()
        params = []

        for i, pe in enumerate(pes.T):

            ac_level = self.ac_level.copy()
            mask = (pe > 10) * (pe < 500) * np.isfinite(pe) * (ac_level >= 0)

            if self.photo_electrons_err is not None:
                err = self.photo_electrons_err[:, i]
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
        self._params = params

    def _interpolate(self):

        from scipy.interpolate import splprep, splev

        pes = self.photo_electrons.copy().T
        cubic_spline = []

        for pe in pes:

            mask = np.isfinite(pe) # * (pe > 5)
            x = self.ac_level[mask]
            y = pe[mask]

            if not len(x):

                x = self.ac_level
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

    test.plot(pixel=1295)

    plt.show()
