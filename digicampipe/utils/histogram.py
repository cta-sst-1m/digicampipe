import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt

print(os.path.realpath(__file__))
lib = np.ctypeslib.load_library("histogram_c.so", os.path.dirname(__file__))
histogram = lib.histogram
histogram.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]

__all__ = ['Histogram1D']


class Histogram1D:

    def __init__(self,  bin_edges, data_shape=(1, ), name="1D Histogram", axis_name=None):

        self.data = np.zeros(data_shape + (bin_edges.shape[0] - 1, ), dtype=np.uint32)
        self.shape = self.data.shape
        self.bins = np.sort(bin_edges).astype(np.float32)
        self.bin_centers = np.diff(self.bins) / 2. + self.bins[:-1]
        self.n_bins = self.bin_centers.shape[-1]
        self.name = name
        self.axis_name = axis_name
        self.underflow = np.zeros(data_shape, dtype=np.uint32)
        self.overflow = np.zeros(data_shape, dtype=np.uint32)

    def fill(self, data_points):

        if data_points.shape[:-1] != self.shape[:-1]:

            raise IndexError

        new_first_axis = 1
        for i in self.shape[:-1]:
            new_first_axis *= i

        data_points = data_points.reshape(new_first_axis, -1).astype(np.float32, order='C')

        self.data = self.data.reshape(new_first_axis, -1)
        self.underflow = self.underflow.reshape(new_first_axis, -1)
        self.overflow = self.overflow.reshape(new_first_axis, -1)

        histogram(data_points, self.data, self.underflow, self.overflow, self.bins, data_points.shape[0], data_points.shape[-1], self.n_bins + 1)

        self.data = self.data.reshape(self.shape)
        self.underflow = self.underflow.reshape(self.shape[:-1])
        self.overflow = self.overflow.reshape(self.shape[:-1])

    def errors(self, index=[...]):

        return np.sqrt(self.data[index])

    def mean(self, index=[...]):

        return np.sum(self.data[index] * self.bin_centers, axis=-1) / np.sum(self.data[index], axis=-1)

    def std(self, index=[...]):

        std = np.sum(self.data[index] * self.bin_centers **2, axis=-1)
        std /= np.sum(self.data[index], axis=-1)
        std -= self.mean(index=index)**2
        return np.sqrt(std)

    def show(self, index, axis=None, normed=False, **kwargs):

        if axis is None:

            fig = plt.figure()
            axis = fig.add_subplot(111)

        text = ' counts : {}\n underflow : {}\n overflow : {}\n mean : {:.4f}\n std : {:.4f}'.format(np.sum(self.data[index]), np.sum(self.underflow[index]), np.sum(self.overflow[index]), self.mean(index=index), self.std(index=index))
        x = self.bin_centers
        y = self.data[index]
        err = self.errors(index=index)
        mask = y > 0

        x = x[mask]
        y = y[mask]
        err = err[mask]

        if normed:

            weights = np.sum(y, axis=-1)
            y = y / weights
            err = err / weights

        steps = axis.step(x, y, where='mid', label='{}'.format(index), **kwargs)
        axis.errorbar(x, y, yerr=err, linestyle='None', color=steps[0].get_color())

        anchored_text = AnchoredText(text, loc=2)
        axis.add_artist(anchored_text)

        axis.set_xlabel('{}'.format(self.axis_name))
        axis.set_ylabel('count' if not normed else 'probability')
        axis.legend(loc='best')


if __name__ == '__main__':

    my_histo = Histogram1D(data_shape=(5, 3), bin_edges=np.arange(-100, 100, 1), axis_name='ADC')
    dat = np.random.normal(0, 5, size=(5, 3,  8000))
    my_histo.fill(data_points=dat)
    dat = np.random.normal(0, 5, size=(5, 3, 10000))
    my_histo.fill(data_points=dat)

    print(np.mean(dat, axis=-1))

    print(my_histo.bin_centers)
    print(my_histo.data)
    print(my_histo.underflow)
    print(my_histo.overflow)
    print(my_histo.mean())
    print(my_histo.std())

    my_histo.show(index=(4, 2), normed=False)
    plt.show()
