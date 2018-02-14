#!/usr/bin/env python
'''

Example:

Usage:
  nsb_evaluation.py [options] <files>...


Options:
  -h --help     Show this screen.
  --display     show plots
  -o <path>, --outfile_path=<path>   path to the output file
  -b <path>, --baseline_path=<path>  path to baseline file usually called "dark.npz"
'''
from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import Camera
from digicampipe.io.save_external_triggers import save_external_triggers
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from docopt import docopt


def main(
    files,
    outfile_path,
    baseline_path,
    do_plots=False
):
    baseline = np.load(baseline_path)

    digicam = Camera()
    data_stream = event_stream(
        file_list=files,
        expert_mode=True,
        camera=digicam,
        camera_geometry=digicam.geometry
    )
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=1050)
    data_stream = filter.filter_missing_baseline(data_stream)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = r1.calibrate_to_r1(data_stream, baseline)
    data_stream = filter.filter_period(data_stream, period=10*u.second)

    save_external_triggers(
        data_stream,
        output_filename=outfile_path,
        pixel_list=np.arange(1296)
    )

    if do_plots:
        make_plots(outfile_path)


def make_plots(path):
    data = np.load(path)
    plot_5(data)
    plot_4(data)
    plot_3(data)
    plot_2(data)
    plot_1(data)
    plt.show()


def plot_5(data):
    plt.figure()
    plt.hist(data['baseline_dark'].ravel(), bins='auto', label='dark')
    plt.hist(data['baseline'].ravel(), bins='auto', label='nsb')
    plt.hist(data['baseline_shift'].ravel(), bins='auto', label='shift')
    plt.xlabel('baseline [LSB]')
    plt.ylabel('count')
    plt.legend()


def plot_4(data):
    digicam = Camera()
    sectors = [1, 3]
    pixel_not_in_intersil = [
        pixel.ID for pixel in digicam.Pixels if pixel.sector in sectors
    ]
    print(len(pixel_not_in_intersil))
    x = data['nsb_rate'].T[pixel_not_in_intersil]

    x = np.mean(x, axis=-1)

    indices_to_keep = np.where(x > 0)
    x = x[indices_to_keep[0]]

    temp = np.where(
        (
            (data['baseline_shift'].T <= 0) +
            (data['baseline_shift'].T >= 80)
        ) > 0
    )
    print(temp[0].shape, temp[1].shape)

    plt.figure()
    plt.hist(temp[0], bins=np.arange(0, 1296, 1))

    plt.figure()
    plt.hist(
        x.ravel(),
        bins='auto',
        label='sector : {} \n mean : {:0.2f} \n std : {:0.2f}'.format(
            sectors,
            np.mean(x),
            np.std(x)
        )
    )
    plt.legend()
    plt.xlabel('$f_{nsb} [GHz]$')


def plot_3(data):
    baseline_max_min = (
        np.max(data['baseline'], axis=0) -
        np.min(data['baseline'], axis=0)
    )
    baseline_max_dark = (
        np.max(data['baseline'], axis=0) -
        np.mean(data['baseline_dark'], axis=0)
    )
    baseline_change = (
        np.diff(data['baseline'], axis=0) /
        np.diff(data['time_stamp'] * 1E-9)[:, np.newaxis]
    )

    mask = baseline_max_min < 25
    mask *= baseline_max_min > 0
    mask *= baseline_max_dark > 55
    mask *= baseline_max_dark < 60
    mask = mask[:, np.newaxis].T * np.abs(baseline_change) < 0.4
    mask *= data['baseline_shift'][:-1] > 30
    mask *= data['baseline_shift'][:-1] < 80

    x = data['nsb_rate'][:-1]
    x = np.ma.array(x, mask=~mask)
    x = np.mean(x, axis=0)
    x = x[np.all(mask, axis=0)]

    plt.figure()
    plt.hist(
        x,
        bins='auto',
        label='mean : {:.2f}, std : {:.2f}'.format(
            np.mean(x),
            np.std(x)
        )
    )
    plt.xlabel('$f_{nsb}$ [GHz]')
    plt.ylabel('count')
    plt.legend()


def plot_2(data):
    plt.figure()
    plt.hist(
        data['nsb_rate'][
            (data['nsb_rate'] > 0.5) * (data['nsb_rate'] < 1.8)
        ].ravel() * 1E3,
        bins='auto',
        label='all pixels'
    )
    plt.xlabel('$f_{nsb}$ [MHz]')
    plt.ylabel('count')
    plt.legend()


def plot_1(data):
    x = np.mean(data['nsb_rate'], axis=0)
    x = x[(x > 0.5) * (x < 1.8)]
    plt.figure()
    plt.hist(
        x,
        bins='auto',
        label='mean : {:.2f}, std : {:.2f}'.format(
            np.mean(x),
            np.std(x)
        )
    )
    plt.xlabel('$f_{nsb}$ [GHz]')
    plt.ylabel('count')
    plt.legend()


if __name__ == '__main__':
    args = docopt(__doc__)
    main(
        files=args['<files>'],
        outfile_path=args['--outfile_path'],
        baseline_path=args['--baseline_path'],
        do_plots=args['--display']
    )
