#!/usr/bin/env python
"""
measure the time offsets using template fitting. Create a file
time_acXXX_dcYYY.npz for eac AC and DC level with XXX and YYY the value of
resp. the AC and DC level.
Usage:
  digicam-time-resolution-plot [options] [--] <INPUT>...

Options:
  -h --help                     Show this screen.
  <INPUT>                       List of path to npz files created by
                                digicam-time-resolution.
  --plot_summary=PATH           Path to the image created showing a summary of
                                the analysis. Set to "none" to not create the
                                plot. Set to "show" to show the plot instead.
                                [Default: none]
  --plot_resolution=PATH        Path to the image created showing the time
                                resolution function of the charge. Set to
                                "none" to not create the plot. Set to "show" to
                                show the plot instead. [Default: show]
  --legend=TEXT                 Legend for the resolution plot.
                                [default: camera average]
  --plot_rms_difference=PATH    Path to the image created showing the time
                                resolution between any combination of 2 pixels
                                with the pulse charge given as
                                "n_pe_rms_difference". Set to "none" to not
                                create the plot. Set to "show" to
                                show the plot instead. [default: none]
  --n_pe_rms_difference=FLOAT   Charge in pe used to calculate the time
                                resolution between any combination of 2 pixels.
                                [default: 5.5]
"""
from docopt import docopt
import os
import numpy as np
from tempfile import TemporaryDirectory
from matplotlib import pyplot as plt
from ctapipe.visualization import CameraDisplay

from digicampipe.utils.docopt import convert_float, convert_text
from digicampipe.instrument.camera import DigiCam
from digicampipe.instrument.light_source import ACLED


cone_pde = 0.88
sipm_pde = 0.35
window_pde = 0.9
pde = cone_pde * sipm_pde * window_pde

plt.rcParams.update({'font.size': 18})


def get_ac_led_light_level(filename_calib='mpe_fit_results_combined.npz'):
    file_calib = os.path.join(filename_calib)
    data_calib = np.load(file_calib)
    ac_led = ACLED(
        data_calib['ac_levels'][:, 0],
        data_calib['mu'],
        data_calib['mu_error']
    )
    return ac_led


def plot_rms_difference(data_file, figure_file, n_pe=20*pde):
    ac_led = get_ac_led_light_level()

    data = np.load(data_file)
    ac_levels = data['ac_levels']
    if 'dc_levels' not in data.keys():
        dc_levels = np.zeros_like(ac_levels)
    else:
        dc_levels = data['dc_levels']
    std_t_all = data['std_t_all']

    # AC LED were calib without DC and without the window
    if np.all(dc_levels == 0):
        print('WARNING: no filter taken into account')
        window_trans = 1
    else:
        window_trans = 0.9
    true_pe = ac_led(ac_levels).T * window_trans

    std_t_npe = np.array(
        [np.interp(n_pe, true_pe[:, i], std_t_all[:, i]) for i in range(1296)]
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    rms_diff = np.sqrt(std_t_npe[:, None]**2 + std_t_npe[None, :]**2)
    triangular = np.tril(np.ones([1296, 1296]), k=-1)
    rms_diff_no_double = rms_diff * triangular
    rms_diff_no_double = rms_diff_no_double[rms_diff_no_double > 0]
    mean = np.mean(rms_diff_no_double)
    std = np.std(rms_diff_no_double)
    print(len(rms_diff_no_double), 'combinations')
    print(np.sum(rms_diff_no_double <= 0), 'pixels without time info')
    ax.hist(
        rms_diff_no_double, 100,
        label='mean={:.2f} ns\nstd={:.2f} ns'.format(mean, std))
    ax.set_xlabel('rms time difference [ns]')
    ax.set_ylabel('# pixel combination')
    ylim = ax.get_ylim()
    ax.plot([2, 2], ylim, 'r-', label='requirement')
    ax.set_ylim(ylim)
    ax.legend()
    plt.tight_layout()
    if figure_file.lower() != "show":
        plt.savefig(figure_file)
        print(figure_file, 'created')
    else:
        plt.show()
    plt.close(fig)


def plot_zone(x, y, bins, ax, label, xscale="log", yscale="linear"):
    H, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
    yy = yedges[:-1] + np.diff(yedges) / 2
    xx = xedges[:-1] + np.diff(xedges) / 2
    mean_y = yy * H
    mean_y = np.sum(mean_y, axis=-1) / np.sum(H, axis=-1)
    sigma_y = ((mean_y[:, None] - yy)) ** 2 * H
    sigma_y = np.sum(sigma_y, axis=-1) / np.sum(H, axis=-1)
    std_y = np.sqrt(sigma_y)
    ax.fill_between(xx, mean_y + std_y, mean_y - std_y, alpha=0.3, color='k',
                    label='$1\sigma$')
    ax.plot(xx, mean_y, color='k', label=label)
    x_min, x_max = bins[0][0], bins[0][-1]
    y_min, y_max = bins[1][0], bins[1][-1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('$N$ [p.e.]')
    ax.grid(True)
    ax.legend()
    # ax2 = ax.twiny()  # instantiate a second axes that shares the same y-axis
    # ax2.plot(1e-5, 1e-5, alpha=0)
    # ax2.tick_params(axis='x')
    # ax2.set_xlim(x_min / pde, x_max / pde)
    # ax2.set_ylim(y_min, y_max)
    # ax2.set_xscale(xscale)
    # ax2.set_yscale(yscale)
    # ax2.xaxis.tick_top()
    # ax2.set_xlabel('$N_\gamma$ [ph.]')
    # ax2.xaxis.set_label_position('top')


def plot_resol(data_file, figure_file, legend):
    ac_led = get_ac_led_light_level()

    data = np.load(data_file)
    ac_levels = data['ac_levels']
    if 'dc_levels' not in data.keys():
        dc_levels = np.zeros_like(ac_levels)
    else:
        dc_levels = data['dc_levels']
    std_t_all = data['std_t_all']

    # AC LED were calib without DC and without the window
    if np.all(dc_levels == 0):
        print('WARNING: no filter taken into account')
        window_trans = 1
    else:
        window_trans = 0.9
    true_pe = ac_led(ac_levels).T * window_trans

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    ax.plot([20*pde, 2e3*pde], [1, 1], 'r-', label='requirement B-TEL-1380')
    ax.plot([20*pde, 20*pde], [.9, 1.1], 'r-', label=None)
    ax.plot([20*pde, 2e3*pde], [1.5, 1.5], 'b--',
            label='requirement B-TEL-1640')
    ax.plot([20*pde, 20*pde], [1.4, 1.6], 'b-', label=None)
    plot_zone(
        true_pe,
        std_t_all,
        [np.logspace(.5, 2.75, 101), np.logspace(-1.2, 1, 101)],
        ax,
        legend,
        yscale='log'
    )
    ax.set_ylabel('time resolution [ns]')

    plt.tight_layout()
    if figure_file.lower() != "show":
        plt.savefig(figure_file)
        print(figure_file, 'created')
    else:
        plt.show()
    plt.close(fig)


def plot_all(data_file, figure_file='time_resolution.png'):
    ac_led = get_ac_led_light_level()

    data = np.load(data_file)
    ac_levels = data['ac_levels']
    if 'dc_levels' not in data.keys():
        dc_levels = np.zeros_like(ac_levels)
    else:
        dc_levels = data['dc_levels']
    mean_charge_all = data['mean_charge_all']
    std_charge_all = data['std_charge_all']
    mean_t_all = data['mean_t_all']
    std_t_all = data['std_t_all']

    # AC LED were calib without DC and without the window
    if np.all(dc_levels == 0):
        print('WARNING: no filter taken into account')
        window_trans = 1
    else:
        window_trans = 0.9
    true_pe = ac_led(ac_levels).T * window_trans
    mean_t_100pe = np.array(
        [np.interp(100, true_pe[:, i], mean_t_all[:, i]) for i in range(1296)]
    )
    std_t_100pe = np.array(
        [np.interp(100, true_pe[:, i], std_t_all[:, i]) for i in range(1296)]
    )

    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    plot_zone(
        true_pe,
        mean_charge_all,
        [np.logspace(-.7, 2.8, 101), np.logspace(-.3, 2.8, 101)],
        axes[0, 0],
        'camera average',
        yscale='log'
    )
    axes[0, 0].loglog([0.1, 1000], [0.1, 1000], 'k--')
    axes[0, 0].set_ylabel('mean charge reco. [p.e]')
    plot_zone(
        true_pe,
        std_charge_all,
        [np.logspace(-.7, 2.8, 101), np.logspace(-0.5, 1.5, 101)],
        axes[0, 1],
        'camera average',
        yscale='log'
    )
    axes[0, 1].loglog([0.1, 1000], np.sqrt([0.1, 1000]), 'k--')
    axes[0, 1].set_ylabel('std charge reco. [p.e]')
    plot_zone(
        true_pe,
        std_t_all,
        [np.logspace(-.7, 2.8, 101), np.logspace(-1.3, 1, 101)],
        axes[0, 2],
        'camera average',
        yscale='log'
    )
    axes[0, 2].set_ylabel('time resolution [ns]')
    plot_zone(
        true_pe,
        mean_t_all - mean_t_100pe[None, :],
        [np.logspace(-.7, 2.8, 101), np.linspace(-2, 2, 101)],
        axes[1, 0],
        'camera average',
        xscale='log',
        yscale='linear'
    )
    axes[1, 0].set_ylabel('time offset [ns]')
    display = CameraDisplay(
        DigiCam.geometry, ax=axes[1, 1],
        title='timing offset (at 100 p.e) [ns]'
    )
    display.image = mean_t_100pe - np.nanmean(mean_t_100pe)
    display.set_limits_minmax(-2, 2)
    display.add_colorbar(ax=axes[1, 1])
    display = CameraDisplay(
        DigiCam.geometry, ax=axes[1, 2],
        title='timing resolution (at 100 p.e.) [ns]'
    )
    display.image = std_t_100pe
    display.set_limits_minmax(0.1, 0.3)
    display.add_colorbar(ax=axes[1, 2])
    plt.tight_layout()
    if figure_file.lower() != "show":
        plt.savefig(figure_file)
        print(figure_file, 'created')
    else:
        plt.show()
    plt.close(fig)


def combine(acdc_level_files, output):
    mean_charge_all = []
    std_charge_all = []
    mean_t_all = []
    std_t_all = []
    ac_levels = []
    dc_levels = []
    n_file = len(acdc_level_files)
    for data_file in acdc_level_files:
        data = np.load(data_file)
        mean_charge_all.append(data['mean_charge'])
        std_charge_all.append(data['std_charge'])
        mean_t_all.append(data['mean_t'])
        std_t_all.append(data['std_t'])
        ac_levels.append(data['ac_level'])
        dc_levels.append(data['dc_level'])
    levels = [list(range(n_file)), dc_levels, ac_levels]
    levels_sorted = sorted(np.array(levels).T, key=lambda x: (x[1], x[2]))
    order = np.array(levels_sorted)[:, 0]
    ac_levels = np.array(ac_levels)[order]
    dc_levels = np.array(dc_levels)[order]
    mean_charge_all = np.array(mean_charge_all)[order]
    std_charge_all = np.array(std_charge_all)[order]
    mean_t_all = np.array(mean_t_all)[order]
    std_t_all = np.array(std_t_all)[order]
    np.savez(
        output,
        ac_levels=ac_levels,
        dc_levels=dc_levels,
        mean_charge_all=mean_charge_all,
        std_charge_all=std_charge_all,
        mean_t_all=mean_t_all,
        std_t_all=std_t_all
    )


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    print('files:', files)
    summary_filename = convert_text(args['--plot_summary'])
    resolution_filename = convert_text(args['--plot_resolution'])
    legend = convert_text(args['--legend'])
    rms_difference_filename = convert_text(args['--plot_rms_difference'])
    n_pe = convert_float(args['--n_pe_rms_difference'])
    with TemporaryDirectory() as temp_dir:
        data_combined = os.path.join(temp_dir, 'time_resolution_test.npz')
        combine(files, data_combined)
        if summary_filename is not None:
            plot_all(data_combined, figure_file=summary_filename)
        if resolution_filename is not None:
            plot_resol(
                data_combined,
                figure_file=resolution_filename,
                legend=legend
            )
        if rms_difference_filename is not None:
            plot_rms_difference(
                data_combined,
                figure_file=rms_difference_filename,
                n_pe=n_pe
            )


if __name__ == '__main__':
    entry()
