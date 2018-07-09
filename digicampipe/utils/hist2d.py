import numpy as np


class Histogram2d:

    def __init__(self, shape, range):
        self.histo = np.zeros(shape, dtype='u2')
        self.range = range
        self.extent = list(self.range[0]) + list(self.range[1])

    def fill(self, x, y):
        for pixel_id in range(len(x)):
            H, xedges, yedges = np.histogram2d(
                x[pixel_id],
                y[pixel_id],
                bins=self.histo.shape[1:],
                range=self.range
            )
            self.histo[pixel_id] += H.astype('u2')

    def contents(self):
        return self.histo


class Histogram2dChunked(Histogram2d):

    def __init__(self, shape, range, buffer_size=1000):
        super().__init__(shape=shape, range=range)

        self.buffer_size = buffer_size
        self.buffer_x = None
        self.buffer_y = None
        self.buffer_counter = 0

    def fill(self, x, y):
        if self.buffer_counter == self.buffer_size:
            self.__fill_histo_from_buffer()

        if self.buffer_x is None:
            self.__reset_buffer(x, y)

        self.buffer_x[self.buffer_counter] = x
        self.buffer_y[self.buffer_counter] = y
        self.buffer_counter += 1

    def __reset_buffer(self, x, y):
        self.buffer_x = np.zeros(
            (self.buffer_size, *x.shape),
            dtype=x.dtype
        )
        self.buffer_y = np.zeros(
            (self.buffer_size, *y.shape),
            dtype=y.dtype
        )
        self.buffer_counter = 0

    def __fill_histo_from_buffer(self):
        if self.buffer_x is None:
            return

        self.buffer_x = self.buffer_x[:self.buffer_counter]
        self.buffer_y = self.buffer_y[:self.buffer_counter]
        for pixel_id in range(self.buffer_x.shape[1]):
            H, xedges, yedges = np.histogram2d(
                self.buffer_x[:, pixel_id].flatten(),
                self.buffer_y[:, pixel_id].flatten(),
                bins=self.histo.shape[1:],
                range=self.range
            )
            self.histo[pixel_id] += H.astype('u2')
        self.buffer_x = None
        self.buffer_y = None
        self.buffer_counter = 0

    def contents(self):
        self.__fill_histo_from_buffer()
        return self.histo
