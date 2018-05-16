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

from ctapipe.visualization.mpl import CameraDisplay
from digicampipe.utils import DigiCam


def fill_histo_from_buffer(histo, buffer):
    for pid in tqdm(range(histo.shape[0])):
        for sid in range(histo.shape[1]):
            histo[pid, sid] += np.bincount(
                buffer[:, pid, sid], minlength=histo.shape[2]
            ).astype(histo.dtype)


def main(outfile_path, input_files=[], chunk_size=10000, plots_path=None):
    events = calibration_event_stream(input_files)

    histo = None
    buffer = None

    for i, e in enumerate(tqdm(events)):
        j = i % chunk_size
        adc = e.data.adc_samples

        if histo is None:
            histo = np.zeros((*adc.shape, 4096), dtype='u2')

        if j == 0:
            if buffer is not None:
                fill_histo_from_buffer(histo, buffer)
            buffer = np.zeros((chunk_size, *adc.shape), dtype=adc.dtype)
        buffer[j] = adc[...]
    buffer = buffer[:j]
    fill_histo_from_buffer(histo, buffer)

    # store buffer zipped to a h5 file
    outfile = h5py.File(outfile_path)
    outfile.create_dataset(
        name='adc_count_histo',
        data=histo,
        compression="gzip",
        chunks=(4, histo.shape[1], 128)
    )

    mean = np.zeros(adc.shape, dtype='f4')
    std = np.zeros(adc.shape, dtype='f4')

    for pid in range(len(adc)):
        D = histo[pid]
        y = np.arange(4096)[np.newaxis, :]
        M = (D * y).sum(axis=1) / D.sum(axis=1)
        mean[pid] = M
        std[pid] = np.sqrt(
            (D * (y - M[:, np.newaxis])**2).sum(axis=1) /
            D.sum(axis=1)
        )

    outfile.create_dataset(
        name='adc_count_mean',
        data=mean,
    )

    outfile.create_dataset(
        name='adc_count_std',
        data=std,
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
        for pixel_id in tqdm(range(1296)):
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
