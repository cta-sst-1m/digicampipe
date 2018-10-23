import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

class Histogram2d:
    def __init__(self, shape, range):
        """
        Create a 2D histogram
        :param shape: tuple describing the shape. The last 2 elements
        correspond to the number of bins in the 1st and 2nd dimension of
        the 2d histograms.
        :param range: 2-tuple (for 1st and 2nd dimension) of
        2-tuples (Lowermost and uppermost limits of the bins)
        """
        self.histo = np.zeros(shape, dtype='u2')
        self.range = range
        self.xedges = None
        self.yedges = None

    def fill(self, x, y):
        for pixel_id in range(len(x)):
            H, xedges, yedges = np.histogram2d(
                x[pixel_id],
                y[pixel_id],
                bins=self.histo.shape[1:],
                range=self.range
            )
            self.histo[pixel_id] += H.astype('u2')
        self.xedges, self.yedges = xedges, yedges

    def contents(self):
        return self.histo

    def fit_y(self, min_entries=2):
        """
        Get the mean and std value for each bin along the 1st dimension
        :param min_entries: minimum number of entries in the bin.
        If there is less entries than that, the bin is skiped.
        :return: (x_bin_centers, means_y, stds_y) with:
        x_bin_centers: center of the bins along the 1st dimension.
        means_y: average values along the second dimension for each
        x_bin_centers
        stds_y: standard deviation along the second dimension for each
        x_bin_centers
        """
        h = self.contents()
        if len(h.shape) == 2:
            h = h.reshape([1, h.shape[0], h.shape[1]])
        x_bin_center = 0.5 * (self.xedges[1:] + self.xedges[:-1])
        y_bin_center = 0.5 * (self.yedges[1:] + self.yedges[:-1])
        x_bin_centers = []
        means_y = []
        stds_y = []
        shape_2d = h.shape[-2:]
        n_2d_hist = sum(h.shape[:-2])
        h_reshaped = h.reshape([n_2d_hist, shape_2d[0], shape_2d[1]])
        n = h_reshaped.sum(axis=-1) #number of entries per bins of the 1st dim
        for h2d_idx in range(n_2d_hist):
            x_bin_non_empty = n[h2d_idx, :] > min_entries
            h_pix = h_reshaped[h2d_idx, x_bin_non_empty, :]
            n_pix = n[h2d_idx, x_bin_non_empty]
            x_bin_centers.append(x_bin_center[x_bin_non_empty])
            mean_y = (h_pix * y_bin_center[None, :]).sum(axis=-1) / n_pix
            means_y.append(mean_y)
            squared_sum_y = (y_bin_center[None, :] - mean_y[:, None]) ** 2
            std_y = np.sqrt((h_pix * squared_sum_y).sum(axis=-1) / (n_pix - 1))
            stds_y.append(std_y)
        return x_bin_centers, means_y, stds_y

    def save(self, path, **kwargs):
        hdu_histo = fits.PrimaryHDU(data=self.contents())
        hdu_range = fits.ImageHDU(data=self.range)
        hdu_xedges = fits.ImageHDU(data=self.xedges)
        hdu_yedges = fits.ImageHDU(data=self.yedges)
        hdul = fits.HDUList([hdu_histo, hdu_range, hdu_xedges, hdu_yedges])
        hdul.writeto(path)

    @classmethod
    def load(cls, path):
        with fits.open(path) as hdul:
            histo = hdul[0].data
            range = hdul[1].data
            obj = Histogram2d(histo.shape, range)
            obj.histo = histo
            obj.xedges = hdul[2].data
            obj.yedges = hdul[3].data
        return obj

    def stack_all(self):
        """
        stack all 2D histograms together and return the result.
        :return: a simple histogram2D
        """
        _h = self.contents()
        shape_2d = _h.shape[-2:]
        hist = Histogram2d(_h.shape[-2:], self.range)
        hist.xedges = self.xedges
        hist.yedges = self.yedges
        n_2d_hist = np.sum(_h.shape[:-2])
        h_reshaped = _h.reshape([n_2d_hist, shape_2d[0], shape_2d[1]])
        for h_2d in h_reshaped:
            hist.histo += h_2d
        return hist

    def plot(self, filename):
        """
        Plot 12 the first non empty 2d histograms.
        :param filename: filename of the plot. if "show", the plot is shown
        instead.
        """
        n_plotted = 0
        plt.figure(figsize=[16, 12], dpi=100)
        fig, axes = plt.subplots(4, 3)
        for i, h in enumerate(self.contents()):
            if np.all(h == 0):
                continue
            ax = axes[int(n_plotted / 3), n_plotted % 3]
            ax.set_title('pixel ' + str(i))
            ax.pcolor(self.xedges, self.yedges, h.T)
            n_plotted += 1
            if n_plotted == 12:
                break
        plt.tight_layout()
        plt.savefig('pulse_shape.png', dpi=200)
        plt.close()


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
        self.xedges, self.yedges = xedges, yedges
        self.buffer_x = None
        self.buffer_y = None
        self.buffer_counter = 0

    def contents(self):
        self.__fill_histo_from_buffer()
        return self.histo
