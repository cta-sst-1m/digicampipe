import matplotlib.pyplot as plt
import numpy as np


class ACLEDInterpolator:

    def __init__(self, ac_level, photoelectrons):

        self.ac_level = ac_level
        self.photoelectrons = photoelectrons
        self._template = self._interpolate()

    def __call__(self, ac_level, pixel=None):

        y = self._template(ac_level)[pixel]

        return y

    def __getitem__(self, item):

        return ACLEDInterpolator(ac_level=self.ac_level,
                                 photoelectrons=self.photoelectrons[item])

    @classmethod
    def load(cls, filename):

        t, x = np.loadtxt(filename).T

        return cls(amplitude=x, time=t)

    def _interpolate(self):

        ac_level = self.ac_level
        pe = self.photoelectrons

        mask = (pe > 1) * (pe < 200)
        pe = np.ma.masked_array(pe, mask=mask)
        param = np.polyfit(ac_level, pe, deg=4)

        return lambda x: np.polyval(param, x)

    def plot(self, axes=None, pixel=None, **kwargs):

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        x_fit = np.arange(1000)
        y_fit = self(x_fit)[pixel]

        axes.plot(self.ac_level, self.photoelectrons, label='Data points',
                  linestyle='None', marker='o', color='k', **kwargs)
        axes.plot(x_fit, y_fit, label='Interpolated template', color='r')
        axes.legend(loc='best')

        return axes
