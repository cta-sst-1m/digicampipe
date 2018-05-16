#!/usr/bin/env python
'''
Reconstruct the pulse template

Usage:
  digicam-template [options] <outfile_path> <input_files>...

Options:
  -h --help               Show this screen.
  --plots=PATH            path where to save a pdf full of interesting images.
  --chunk_size INT        how many events to read before histogramming.
                          This uses about 1GB memory, if you have less mem,
                          reduce this number (will make the program slower).
                          [default: 10000]
'''
import numpy as np
import h5py
from digicampipe.io.event_stream import calibration_event_stream

from tqdm import tqdm
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ctapipe.visualization import CameraDisplay
from digicampipe.utils import DigiCam
import numba


class Histogram2D:

    def __init__(self, shape, dtype, chunk_size=10000):
        self.shape = shape
        self.histo = np.zeros(shape, dtype=dtype)
        self._buffer = None
        self.chunk_size = chunk_size
        self.buffer_index = 0
        self._mean = None
        self._std = None

    def fill(self, data):
        if self._buffer is None:
            self._buffer = np.zeros(
                (self.chunk_size, *data.shape),
                dtype=data.dtype
            )

        self._buffer[self.buffer_index] = data[...]
        self.buffer_index += 1

        if self.buffer_index == self.chunk_size:
            self._fill_histo_from_buffer()

    def _fill_histo_from_buffer(self):
        buffer = self._buffer[:self.buffer_index]

        n_pixel = self.histo.shape[0]
        n_samples = self.histo.shape[1]
        histo_height = self.histo.shape[2]
        for pixel_id in range(n_pixel):
            for sample_id in range(n_samples):
                self.histo[pixel_id, sample_id] += np.bincount(
                    buffer[:, pixel_id, sample_id],
                    minlength=histo_height
                ).astype(self.histo.dtype)

        self._buffer = None
        self.buffer_index = 0

    def contents(self):
        if self._buffer is not None:
            self._fill_histo_from_buffer()
        return self.histo

    def calc_stuff(self):
        self._fill_histo_from_buffer()
        self._mean = np.zeros(
            self.histo.shape[:-1],
            dtype='f4'
        )
        self._std = np.zeros(
            self.histo.shape[:-1],
            dtype='f4'
        )

        for pid in range(self.histo.shape[0]):
            D = self.histo[pid]
            y = np.arange(4096)[np.newaxis, :]
            M = (D * y).sum(axis=1) / D.sum(axis=1)
            self._mean[pid] = M

            self._std[pid] = np.sqrt(
                (D * (y - M[:, np.newaxis])**2).sum(axis=1) /
                D.sum(axis=1)
            )

    def mean(self):
        if self._mean is None:
            self.calc_stuff()
        return self._mean

    def std(self):
        if self._std is None:
            self.calc_stuff()
        return self._std


@numba.jit
def estimate_arrival_time(adc, thr=0.5):
    '''
    estimate the pulse arrival time, defined as the time the leading edge
    crossed 50% of the maximal height,
    estimated using a simple linear interpolation.

    *Note*
    This method breaks down for very small pulses where noise affects the
    leading edge significantly.
    Typical pixels have a ~2LSB electronics noise level.
    Assuming a leading edge length of 4 samples
    then for a typical pixel, pulses of 40LSB (roughly 7 p.e.)
    should be fine.

    adc: (1296, 50) dtype=uint16 or so
    thr: threshold, 50% by default ... can be played with.

    return:
        arrival_time (1296) in units of time_slices
    '''
    n_pixel = adc.shape[0]
    arrival_times = np.zeros(n_pixel, dtype='f4')

    for pixel_id in range(n_pixel):
        y = adc[pixel_id]
        am = y.argmax()
        y_ = y[:am+1]
        lim = y_[-1] * thr
        foo = np.where(y_ < lim)[0]
        if len(foo):
            start = foo[-1]
            stop = start + 1
            arrival_times[pixel_id] = start + (
                (lim-y_[start]) / (y_[stop]-y_[start])
            )
        else:
            arrival_times[pixel_id] = np.nan
    return arrival_times


def main(outfile_path, input_files=[], chunk_size=10000, plots_path=None):
    events = calibration_event_stream(input_files)

    histo = None

    for e in tqdm(events):
        adc = e.data.adc_samples

        if histo is None:
            histo = Histogram2D(
                shape=(*adc.shape, 4096),
                dtype='u2',
                chunk_size=chunk_size
            )

        histo.fill(adc)

    # store buffer zipped to a h5 file
    outfile = h5py.File(outfile_path)
    outfile.create_dataset(
        name='adc_count_histo',
        data=histo.contents(),
        compression="gzip",
        chunks=(4, histo.shape[1], 128)
    )

    outfile.create_dataset(
        name='adc_count_mean',
        data=histo.mean(),
    )

    outfile.create_dataset(
        name='adc_count_std',
        data=histo.std(),
    )

    if plots_path is not None:
        make_plots(outfile, plots_path)


def make_plots(outfile, plots_path, pixel_per_page=(3, 2), plots_per_ax=9):
    f = outfile
    print('making plots into:', plots_path)
    with PdfPages(plots_path) as pdf:

        M = f['adc_count_mean'][...]
        d = M - M[:, 0][:, None]
        trigger = d[:, :-4] - d[:, 4:]
        slope_over_five = trigger[:, :-5] - trigger[:, 5:]

        fig, axes = plt.subplots(2, 2, figsize=(8, 11))
        d = CameraDisplay(
            geometry=DigiCam.geometry,
            image=M.max(axis=1),
            ax=axes[0, 0]
        )
        plt.colorbar(d.pixels, ax=axes[0, 0])
        axes[0, 0].set_title('Maximum heigth of avera pulse shape')
        axes[0, 1].hist(M.max(axis=1), bins='auto', histtype='step', lw=3)

        d = CameraDisplay(
            geometry=DigiCam.geometry,
            image=M.argmax(axis=1),
            ax=axes[1, 0]
        )
        plt.colorbar(d.pixels, ax=axes[1, 0])
        axes[1, 0].set_title('Time of Maximum of average pulse shape')
        axes[1, 1].hist(
            M.argmax(axis=1),
            bins=50,
            histtype='step',
            lw=3
        )
        pdf.savefig()
        plt.close('all')

        fig, axes = plt.subplots(2, 2, figsize=(8, 11))
        d = CameraDisplay(
            geometry=DigiCam.geometry,
            image=slope_over_five.argmax(axis=1) + 5,
            ax=axes[0, 0]
        )
        plt.colorbar(d.pixels, ax=axes[0, 0])
        axes[0, 0].set_title('Time of highest slope')
        axes[0, 1].hist(
            slope_over_five.argmax(axis=1) + 5,
            bins=50,
            histtype='step',
            lw=3
        )

        d = CameraDisplay(
            geometry=DigiCam.geometry,
            image=(M - M[0]).sum(axis=1),
            ax=axes[1, 0]
        )
        plt.colorbar(d.pixels, ax=axes[1, 0])
        axes[1, 0].set_title('area under average pulse shape')
        axes[1, 1].hist(
            (M - M[0]).sum(axis=1),
            bins='auto',
            histtype='step',
            lw=3
        )

        pdf.savefig()
        plt.close('all')

        fig, axes = None, None

        plots_per_page = np.prod(pixel_per_page)
        print('plots_per_page', plots_per_page)
        for pixel_id in tqdm(range(f['adc_count_mean'].shape[0])):
            if pixel_id % (plots_per_page * plots_per_ax) == 0:
                if fig is not None:
                    pdf.savefig()
                    plt.close('all')
                fig, axes = plt.subplots(
                    *pixel_per_page,
                    figsize=(8, 11),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )
                ax = axes.flat

            A = ax[(pixel_id//plots_per_ax) % plots_per_page]

            mean = f['adc_count_mean'][pixel_id]
            std = f['adc_count_std'][pixel_id]
            x = np.arange(len(mean))

            A.errorbar(
                x=x,
                y=mean,
                yerr=std,
                fmt='.:',
                label='pixel_id:{}'.format(pixel_id)
            )
            A.set_xlim(0, 50)
            A.set_ylim(100, 1300)

            A.grid()
            A.legend(loc='upper right')
        pdf.savefig()
        plt.close('all')


def entry():
    args = docopt(__doc__)

    main(
        outfile_path=args['<outfile_path>'],
        input_files=args['<input_files>'],
        chunk_size=int(args['--chunk_size']),
        plots_path=args['--plots']
    )

if __name__ == '__main__':
    entry()
