#!/usr/bin/env python
"""
Plot the output of the pipeline.

Usage:
  digicam-plot-pipeline [options] [--] <INPUT>

Options:
  -h --help                     Show this screen.
  <INPUT>                       Intput file (output from digicam-pipeline).
                                [Default: ./hillas.fits]
  --plot_scan2d=PATH            path to the plot for a 2d scan of the source
                                position for the number of shower with alpha <
                                --alphas_min. If set to "none", the plot is not
                                produced. If set to "show" the plot is
                                displayed instead. [default: none]
  --alphas_min=LIST             Minimum alpha angles in degrees that an event
                                must have during the 2D scan to be included.
                                [Default: 0.2,0.5,1,2,5]
  --plot_map_disp=PATH          path to the plot for a 2d map using the disp
                                method using the --xis factor.
                                If set to "none", the plot is not
                                produced. If set to "show" the plot is
                                displayed instead. [default: none]
  --xis=LIST                    xis parameters used for the map using disp
                                method.
                                [Default: 1,1.2,1.4,1.6,1.8,2]
  --plot_showers_center=PATH    path to the plot of a 2d histogram of shower
                                center of gravity. If set to "none", the plot
                                is not produced. If set to "show" the plot is
                                displayed instead. [default:none]
  --plot_hillas=PATH            path to the plot of the histograms of Hillas
                                parameters. If set to "none", the plot is not
                                produced. If set to "show" the plot is
                                displayed instead. [default:none]
  --plot_correl_all=PATH        path to the plot of a few correlation between
                                Hillas parameter for event without cuts.
                                If set to "none", the plot
                                is not produced. If set to "show" the plot is
                                displayed instead. [default:none]
  --plot_correl_selected=PATH   path to the plot of a few correlation between
                                Hillas parameter for event passing the cuts.
                                If set to "none", the plot
                                is not produced. If set to "show" the plot is
                                displayed instead. [default:none]
  --plot_correl_cut=PATH        path to the plot of a few correlation between
                                Hillas parameter for event not passing the
                                cuts.
                                If set to "none", the plot
                                is not produced. If set to "show" the plot is
                                displayed instead. [default:none]
  --disable_bar                 Disable the progress bar
"""
import numpy as np
import pandas as pd
from astropy.table import Table
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

from digicampipe.utils.docopt import convert_text, convert_list_float
from digicampipe.image.hillas import correct_hillas, compute_alpha, \
    arrival_lessard


def correlation_plot(pipeline_data, title=None, plot="show"):
    fig = plt.figure(figsize=(24, 12))
    subplot = 0
    for i, (label_x, x) in enumerate(zip(
            ['shower center X [mm]', 'shower center Y [mm]'],
            [pipeline_data['x'], pipeline_data['y']]
    )):
        for j, (label_y, y, ymin, ymax) in enumerate(zip(
                [
                    'shower length [mm]',
                    'shower width [mm]',
                    'length/width',
                    'r'
                ],
                [
                    pipeline_data['length'],
                    pipeline_data['width'],
                    pipeline_data['length'] / pipeline_data['width'],
                    pipeline_data['r']
                ],
                [0, 0, 0, -100],
                [200, 100, 10, 500]
        )):
            subplot += 1
            plt.subplot(2, 4, subplot)
            plt.hist2d(x, y, bins=(100, np.linspace(ymin, ymax, 100)),
                       norm=LogNorm())
            plt.ylim(ymin, ymax)
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.title(title)
            cb = plt.colorbar()
            cb.set_label('Number of events')
    plt.tight_layout()
    if plot == "show":
        plt.show()
    else:
        plt.savefig(plot)
        print(plot, 'created')
    plt.close(fig)


def showers_center_plot(pipeline_data, selection, plot="show"):
    # 2d histogram of shower centers
    fig = plt.figure(figsize=(16, 16))

    plt.subplot(2, 2, 1)
    plt.hist2d(pipeline_data['x'], pipeline_data['y'],
               bins=100, norm=LogNorm())
    plt.ylabel('shower center Y [mm]')
    plt.xlabel('shower center X [mm]')
    plt.title('all events')
    cb = plt.colorbar()
    cb.set_label('Number of events')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    data_ok = pipeline_data[pipeline_data['burst']]
    if len(data_ok) > 0:
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('burst')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')

    plt.subplot(2, 2, 3)
    data_ok = pipeline_data[~selection]
    if len(data_ok) > 0:
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('failing cuts')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')

    plt.subplot(2, 2, 4)
    data_ok = pipeline_data[(~pipeline_data['burst']) & selection]
    if len(data_ok) > 0:
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('pass all')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
    plt.tight_layout()
    if plot == "show":
        plt.show()
    else:
        plt.savefig(plot)
        print(plot, 'created')
    plt.close(fig)


def hillas_plot(pipeline_data, selection, plot="show", yscale='log'):
    fig = plt.figure(figsize=(25, 20))
    subplot = 0
    for key, val in pipeline_data.items():
        if key in ['border', 'kurtosis', 'event_id',
                   'event_type', 'burst', 'saturated', 'shower',
                   'pointing_leds_on', 'pointing_leds_blink', 'all_hv_on',
                   'all_ghv_on', 'is_on_source', 'is_tracking',
                   'digicam_temperature']:
            continue
        subplot += 1
        print(subplot, '/', 20, 'plotting', key)
        plt.subplot(4, 5, subplot)
        val_split = [
            val[selection],
            val[~selection]
        ]
        if key == 'intensity':
            plt.xscale('log')
            binmin = np.floor(np.log10(np.nanmin(val)))
            binmax = np.ceil(np.log10(np.nanmax(val)))
            bins = np.logspace(binmin, binmax, 100)
            h, bins, p = plt.hist(val_split, bins=bins, stacked=True)
        else:
            binmin = np.floor(np.nanmin(val))
            binmax = np.ceil(np.nanmax(val))
            if key == 'skewness':
                binmin = -2
                binmax = 2
            if key == 'nsb_rate':
                binmin = np.max(binmin, 0)
                binmax = 3
            if key == 'alpha':
                binmin = 0
                binmax = np.pi / 2
            if key == 'psi':
                binmin = 0
                binmax = np.pi
            if key == 'phi':
                binmin = 0
                binmax = np.pi
            bins = np.linspace(binmin, binmax, 100)
            h, bins, p = plt.hist(val_split, bins=bins, stacked=True)
        plt.xlabel(key)
        plt.yscale(yscale)
        ymax = np.nanmax(h)
        if yscale == 'log':
            plt.ylim([0.8, ymax * 1.5])
        else:
            plt.ylim([0, ymax * 1.1])
        if subplot == 1:
            plt.legend(['pass cuts', 'fail cuts'])
    plt.tight_layout()
    if plot == "show":
        plt.show()
    else:
        plt.savefig(plot)
        print(plot, 'created')
    plt.close(fig)


def scan_2d_plot(
        pipeline_data,
        alphas_min=(1, 2, 5, 10, 20),
        plot="show",
        num_steps=200,
        fov=((-500, 500), (-500, 500)),
        disable_bar=True,
):
    """
    2D scan of spike in alpha
    :param pipeline_data: hillas parameters data file, output of pipeline.py
    :param alphas_min: list of float, each element being a alpha_min.
    events are taken into account as possibly coming from
    the scanned point if the alpha parameter calculated at that point is below
    alpha_min.
    :param plot: path to the plot for a 2d scan of the source position.
    If set to "none", the plot is not produced. If set to "show" the plot
    is displayed instead.
    :param num_steps: number of binning in the FoV
    :param fov: x and y range of the field of view. Format:
    ((x_min, x_max), (y_min, y_max))
    :param disable_bar: If true shows the progress bar of the alpha computation
    :return: None
    """

    alphas_min = np.array(alphas_min)
    x_fov_start = fov[0][0]  # limits of the FoV in mm
    y_fov_start = fov[1][0]  # limits of the FoV in mm
    x_fov_end = fov[0][1]  # limits of the FoV in mm
    y_fov_end = fov[1][1]  # limits of the FoV in mm
    x_fov = np.linspace(x_fov_start, x_fov_end, num_steps)
    y_fov = np.linspace(y_fov_start, y_fov_end, num_steps)
    dx = x_fov[1] - x_fov[0]
    dy = y_fov[1] - y_fov[0]
    x_fov_bins = np.linspace(x_fov_start - dx / 2, x_fov_end + dx / 2,
                             num_steps + 1)
    y_fov_bins = np.linspace(y_fov_start - dy / 2, y_fov_end + dy / 2,
                             num_steps + 1)
    num_alpha = len(alphas_min)
    n = np.zeros([num_steps, num_steps, num_alpha], dtype=int)
    print('2D scan calculation:')

    X, Y = np.meshgrid(x_fov, y_fov)

    for index, hillas in tqdm(pipeline_data.iterrows(),
                              total=len(pipeline_data), disable=disable_bar):

        x, y, r, phi = correct_hillas(hillas['x'], hillas['y'],
                                      source_x=X,
                                      source_y=Y)

        alpha = compute_alpha(phi, hillas['psi'])
        alpha = np.rad2deg(alpha)
        alpha = alpha[..., None] < alphas_min
        n += alpha

    for ai, alpha_min in enumerate(alphas_min):
        if len(alphas_min) > 1:
            plot_name = plot.replace('.png', '_{}deg.png'.format(alpha_min))
        else:
            plot_name = plot
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(111)
        pcm = ax1.pcolormesh(x_fov_bins, y_fov_bins, n[:, :, ai],
                             rasterized=True, cmap='nipy_spectral')
        plt.ylabel('FOV Y [mm]')
        plt.xlabel('FOV X [mm]')
        cbar = fig.colorbar(pcm)
        cbar.set_label('# of events')
        plt.grid()
        plt.tight_layout()
        if plot == "show":
            plt.show()
        else:
            plt.savefig(plot_name)
            print(plot_name, 'created')
        plt.close(fig)


def map_disp(pipeline_data, xis, plot='show', num_steps=(20, 20),
             fov=((-500, 500), (-500, 500))):
    x_fov_start = fov[0][0]  # limits of the FoV in mm
    y_fov_start = fov[1][0]  # limits of the FoV in mm
    x_fov_end = fov[0][1]  # limits of the FoV in mm
    y_fov_end = fov[1][1]  # limits of the FoV in mm
    num_steps = np.atleast_1d(num_steps)
    xis = np.unique(np.abs(np.array(xis)))
    if len(num_steps) == 1:
        num_steps = [num_steps[0], num_steps[0]]
    x_bin = np.linspace(x_fov_start, x_fov_end, num_steps[0] + 1)
    y_bin = np.linspace(y_fov_start, y_fov_end, num_steps[1] + 1)
    x_fov = 0.5 * (x_bin[1:] + x_bin[:-1])
    y_fov = 0.5 * (y_bin[1:] + y_bin[:-1])
    x_cen = 0.5 * (x_fov_start + x_fov_end)
    y_cen = 0.5 * (y_fov_start + y_fov_end)
    bkg_mask = pipeline_data['skewness'] < 0
    data_mask = pipeline_data['skewness'] > 0
    x, y = arrival_lessard(pipeline_data[data_mask], xis)
    x_bkb, y_bkb = arrival_lessard(pipeline_data[bkg_mask], -xis)
    for i, xi in enumerate(np.unique(xis)):
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 5)
        ax1 = plt.subplot(gs[:3, :3])
        h_bkg, _, _ = np.histogram2d(x_bkb[:, i], y_bkb[:, i],
                                     bins=(x_bin, y_bin))
        h, _, _ = np.histogram2d(x[:, i], y[:, i],
                                     bins=(x_bin, y_bin))
        img = ax1.pcolormesh(x_bin, y_bin, (h - h_bkg).T)
        data_center = np.logical_and(np.abs(x[:, i] - x_cen) < 10,
                                     np.abs(y[:, i] - y_cen) < 10)
        # plt.plot(pipeline_data['x'][data_mask][data_center],
        #          pipeline_data['y'][data_mask][data_center], 'r+', ms=10)
        plt.grid()
        ax5 = plt.subplot(gs[3, 3:])
        plotted = pipeline_data['width']  # pipeline_data['length']/pipeline_data['width']
        ax5.hist(plotted[data_mask][data_center][pipeline_data['skewness'] > 0], facecolor='b')
        ax5.hist(plotted[data_mask][data_center][pipeline_data['skewness'] < 0], facecolor='r')
        # colorbar
        ax4 = plt.subplot(gs[:3, 4])
        plt.colorbar(img, cax=ax4)
        # projection on Y axis
        ax2 = plt.subplot(gs[:3, 3], sharey=ax1)
        ax2.plot(np.sum(h, axis=0), y_fov, label='data')
        ax2.plot(np.sum(h_bkg, axis=0), y_fov, label='bkg')
        ax2.plot(np.sum(h-h_bkg, axis=0), y_fov, label='excess')
        plt.setp(ax2.get_yticklabels(), visible=False)
        # projection on X axis
        ax3 = plt.subplot(gs[3, :3], sharex=ax1)
        ax3.plot(x_fov, np.sum(h, axis=1), label='data')
        ax3.plot(x_fov, np.sum(h_bkg, axis=1), label='bkg')
        ax3.plot(x_fov, np.sum(h-h_bkg, axis=1), label='excess')
        ax3.legend()
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.tight_layout()
        if len(xis)> 1:
            plot_name = plot.replace('.png', '_xi{:.2f}.png'.format(xi))
        else:
            plot_name = plot
        if plot == "show":
            plt.show()
        else:
            plt.savefig(plot_name)
            print(plot_name, 'created')
        plt.close(fig)


def cut_data(
        pipeline_data,
        cut_length_gte=None,
        cut_length_lte=None,
        cut_width_gte=None,
        cut_width_lte=None,
        cut_length_over_width_gte=None,
        cut_length_over_width_lte=None,
        cut_intensity_gte=None,
        cut_intensity_lte=None,
        cut_skewness_gte=None,
        cut_skewness_lte=None,
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None,
        cut_led_on_eq=None,
        cut_led_blink_eq=None,
        cut_target_ra_gte=None,
        cut_target_ra_lte=None,
        cut_target_dec_gte=None,
        cut_target_dec_lte=None,
        cut_nsb_rate_gte=None,
        cut_nsb_rate_lte=None,
        cut_r_gte=None,
        cut_r_lte=None,
        cut_n_island_gte=None,
):
    selection = np.isfinite(pipeline_data['intensity'])
    if cut_length_gte is not None:
        event_pass = pipeline_data['length'] < cut_length_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: length < ', cut_length_gte)
    if cut_length_lte is not None:
        event_pass = pipeline_data['length'] > cut_length_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: length > ', cut_length_lte)
    if cut_width_gte is not None:
        event_pass = pipeline_data['width'] < cut_width_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: width < ', cut_width_gte)
    if cut_width_lte is not None:
        event_pass = pipeline_data['width'] > cut_width_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: width > ', cut_width_lte)
    if cut_length_over_width_gte is not None:
        length_over_width = pipeline_data['length'] / pipeline_data['width']
        event_pass = length_over_width < cut_length_over_width_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: l/w < ', cut_length_over_width_gte)
    if cut_length_over_width_lte is not None:
        length_over_width = pipeline_data['length'] / pipeline_data['width']
        event_pass = length_over_width > cut_length_over_width_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: l/w > ',
              cut_length_over_width_lte)
    if cut_intensity_gte is not None:
        event_pass = pipeline_data['intensity'] < cut_intensity_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: intensity < ',
              cut_intensity_gte)
    if cut_intensity_lte is not None:
        event_pass = pipeline_data['intensity'] > cut_intensity_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: intensity > ',
              cut_intensity_lte)
    if cut_skewness_gte is not None:
        event_pass = pipeline_data['skewness'] < cut_skewness_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: skewness < ',
              cut_skewness_gte)
    if cut_skewness_lte is not None:
        event_pass = pipeline_data['skewness'] > cut_skewness_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: skewness > ',
              cut_skewness_lte)
    if cut_border_eq is not None:
        event_pass = pipeline_data['border'] != cut_border_eq
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: border !=', cut_border_eq)
    if cut_burst_eq is not None:
        event_pass = pipeline_data['burst'] != cut_burst_eq
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: burst !=', cut_burst_eq)
    if cut_saturated_eq is not None:
        event_pass = pipeline_data['saturated'] != cut_saturated_eq
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: saturated !=', cut_saturated_eq)
    if cut_led_on_eq is not None:
        event_pass = pipeline_data['pointing_leds_on'] != cut_led_on_eq
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: LEDs on !=', cut_led_on_eq)
    if cut_led_blink_eq is not None:
        event_pass = pipeline_data['pointing_leds_blink'] != cut_led_blink_eq
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: LEDs blink !=', cut_led_blink_eq)
    if cut_target_ra_gte is not None:
        event_pass = pipeline_data['target_ra'] < cut_target_ra_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: target\'s ra < ',
              cut_target_ra_gte)
    if cut_target_ra_lte is not None:
        event_pass = pipeline_data['target_ra'] > cut_target_ra_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: target\'s ra > ',
              cut_target_ra_lte)
    if cut_target_dec_gte is not None:
        event_pass = pipeline_data['target_dec'] < cut_target_dec_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: target\'s dec < ',
              cut_target_dec_gte)
    if cut_target_dec_lte is not None:
        event_pass = pipeline_data['target_dec'] > cut_target_dec_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: target\'s dec > ',
              cut_target_dec_lte)
    if cut_nsb_rate_gte is not None:
        event_pass = pipeline_data['nsb_rate'] < cut_nsb_rate_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: nsb rate < ',
              cut_nsb_rate_gte, 'GHz')
    if cut_nsb_rate_lte is not None:
        event_pass = pipeline_data['nsb_rate'] > cut_nsb_rate_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: nsb rate > ',
              cut_nsb_rate_lte, 'GHz')
    if cut_r_gte is not None:
        event_pass = pipeline_data['r'] < cut_r_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: r < ',
              cut_nsb_rate_gte, 'mm')
    if cut_r_lte is not None:
        event_pass = pipeline_data['r'] > cut_r_lte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: r > ',
              cut_nsb_rate_lte, 'mm')
    if cut_n_island_gte is not None:
        event_pass = pipeline_data['number_of_island'] > cut_n_island_gte
        old_selection = selection
        selection = np.logical_and(selection, event_pass)
        print(np.sum(selection), '/', np.sum(old_selection),
              'events cut with selection: n_island < ',
              cut_n_island_gte)
    return selection


def get_data_and_selection(
        hillas_file,
        cut_length_gte=None,
        cut_length_lte=None,
        cut_width_gte=None,
        cut_width_lte=None,
        cut_length_over_width_gte=None,
        cut_length_over_width_lte=None,
        cut_intensity_gte=None,
        cut_intensity_lte=None,
        cut_skewness_gte=None,
        cut_skewness_lte=None,
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None,
        cut_led_on_eq=None,
        cut_led_blink_eq=None,
        cut_target_ra_gte=None,
        cut_target_ra_lte=None,
        cut_target_dec_gte=None,
        cut_target_dec_lte=None,
        cut_nsb_rate_gte=None,
        cut_nsb_rate_lte=None,
        cut_r_gte=None,
        cut_r_lte=None,
        cut_n_island_gte=None,
):
    data = Table.read(hillas_file, format='fits')
    data = data.to_pandas()
    data['time'] = pd.to_datetime(data['local_time'])
    data = data.set_index('time')
    data = data.dropna()

    selection = cut_data(
        pipeline_data=data,
        cut_length_gte=cut_length_gte,
        cut_length_lte=cut_length_lte,
        cut_width_gte=cut_width_gte,
        cut_width_lte=cut_width_lte,
        cut_length_over_width_gte=cut_length_over_width_gte,
        cut_length_over_width_lte=cut_length_over_width_lte,
        cut_intensity_gte=cut_intensity_gte,
        cut_intensity_lte=cut_intensity_lte,
        cut_skewness_gte=cut_skewness_gte,
        cut_skewness_lte=cut_skewness_lte,
        cut_border_eq=cut_border_eq,
        cut_burst_eq=cut_burst_eq,
        cut_saturated_eq=cut_saturated_eq,
        cut_led_on_eq=cut_led_on_eq,
        cut_led_blink_eq=cut_led_blink_eq,
        cut_target_ra_gte=cut_target_ra_gte,
        cut_target_ra_lte=cut_target_ra_lte,
        cut_target_dec_gte=cut_target_dec_gte,
        cut_target_dec_lte=cut_target_dec_lte,
        cut_nsb_rate_gte=cut_nsb_rate_gte,
        cut_nsb_rate_lte=cut_nsb_rate_lte,
        cut_r_gte=cut_r_gte,
        cut_r_lte=cut_r_lte,
        cut_n_island_gte=cut_n_island_gte,
    )
    return data, selection


def plot_pipeline(
        hillas_file,
        cut_length_gte=None,
        cut_length_lte=None,
        cut_width_gte=None,
        cut_width_lte=None,
        cut_length_over_width_gte=None,
        cut_length_over_width_lte=None,
        cut_intensity_gte=None,
        cut_intensity_lte=None,
        cut_skewness_gte=None,
        cut_skewness_lte=None,
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None,
        cut_led_on_eq=None,
        cut_led_blink_eq=None,
        cut_target_ra_gte=None,
        cut_target_ra_lte=None,
        cut_target_dec_gte=None,
        cut_target_dec_lte=None,
        cut_nsb_rate_gte=None,
        cut_nsb_rate_lte=None,
        cut_r_gte=None,
        cut_r_lte=None,
        cut_n_island_gte=None,
        alphas_min=(1, 2, 5, 10, 20),
        plot_scan2d=None,
        plot_showers_center=None,
        plot_hillas=None,
        plot_correlation_all=None,
        plot_correlation_selected=None,
        plot_correlation_cut=None,
        plot_map_disp=None,
        xis=(1, 1.2, 1.4, 1.6, 1.8, 2.0),
        print_events=0,
        disable_bar=True,
):
    data, selection = get_data_and_selection(
        hillas_file=hillas_file,
        cut_length_gte=cut_length_gte,
        cut_length_lte=cut_length_lte,
        cut_width_gte=cut_width_gte,
        cut_width_lte=cut_width_lte,
        cut_length_over_width_gte=cut_length_over_width_gte,
        cut_length_over_width_lte=cut_length_over_width_lte,
        cut_intensity_gte=cut_intensity_gte,
        cut_intensity_lte=cut_intensity_lte,
        cut_skewness_gte=cut_skewness_gte,
        cut_skewness_lte=cut_skewness_lte,
        cut_border_eq=cut_border_eq,
        cut_burst_eq=cut_burst_eq,
        cut_saturated_eq=cut_saturated_eq,
        cut_led_on_eq=cut_led_on_eq,
        cut_led_blink_eq=cut_led_blink_eq,
        cut_target_ra_gte=cut_target_ra_gte,
        cut_target_ra_lte=cut_target_ra_lte,
        cut_target_dec_gte=cut_target_dec_gte,
        cut_target_dec_lte=cut_target_dec_lte,
        cut_nsb_rate_gte=cut_nsb_rate_gte,
        cut_nsb_rate_lte=cut_nsb_rate_lte,
        cut_r_gte=cut_r_gte,
        cut_r_lte=cut_r_lte,
        cut_n_island_gte=cut_n_island_gte,
    )
    selection_no_burst = np.logical_and(selection, data['burst'] == False)

    if plot_hillas is not None:
        hillas_plot(pipeline_data=data, selection=selection, plot=plot_hillas)
    if plot_showers_center:
        showers_center_plot(pipeline_data=data, selection=selection,
                            plot=plot_showers_center)
    if plot_correlation_all is not None:
        correlation_plot(data, title='all', plot=plot_correlation_all)
    if plot_correlation_selected is not None:
        correlation_plot(data[selection], title='pass cuts',
                         plot=plot_correlation_selected)
    if plot_correlation_cut is not None:
        correlation_plot(data[~selection], title='fail cuts',
                         plot=plot_correlation_cut)
    if plot_scan2d is not None:
        scan_2d_plot(
            pipeline_data=data[selection_no_burst], alphas_min=alphas_min,
            plot=plot_scan2d, disable_bar=disable_bar,
            fov=((-400, 400), (-400, 400)), num_steps=400
        )
    if plot_map_disp is not None:
        map_disp(
            pipeline_data=data[selection_no_burst], xis=xis,
            plot=plot_map_disp,
            fov=((-400, 400), (-400, 400)), num_steps=(51, 51)
        )
    if not np.isfinite(print_events):
        print_events = len(data[selection])
    for event in range(print_events):
        print(
            'event', event, ':',
            't={}'.format(data[selection].index[event]),
            'id={}'.format(data[selection]['event_id'][event]),
            'alpha={:.2f}'.format(data[selection]['alpha'][event]),
            '(x,y)=({:.1f}, {:.1f})'.format(data[selection]['x'][event],
                                            data[selection]['y'][event])
        )


def entry():
    args = docopt(__doc__)
    hillas_file = args['<INPUT>']
    alphas_min = convert_list_float(args['--alphas_min'])
    plot_scan2d = convert_text(args['--plot_scan2d'])
    plot_showers_center = convert_text(args['--plot_showers_center'])
    plot_hillas = convert_text(args['--plot_hillas'])
    plot_correlation_all = convert_text(args['--plot_correl_all'])
    plot_correlation_selected = convert_text(args['--plot_correl_selected'])
    plot_correlation_cut = convert_text(args['--plot_correl_cut'])
    disable_bar = args['--disable_bar']
    xis = convert_list_float(args['--xis'])
    plot_map_disp = convert_text(args['--plot_map_disp'])
    plot_pipeline(
        hillas_file=hillas_file,
        cut_length_gte=None,  # Whipple:43 # 2017:None
        cut_length_lte=None,  # Whipple:16 # 2017:None
        cut_width_gte=None,  # Whipple:16 # 2017:None
        cut_width_lte=None,  # Whipple:7.3 # 2017:None
        cut_length_over_width_gte=3,  # Whipple:None # 2017:3
        cut_length_over_width_lte=1.5,  # Whipple:None # 2017:1.5
        cut_intensity_gte=None,
        cut_intensity_lte=100,  # Whipple:None # 2017: 100?
        cut_skewness_gte=None,
        cut_skewness_lte=None,
        cut_border_eq=True,
        cut_burst_eq=True,
        cut_saturated_eq=None,
        cut_led_on_eq=True,
        cut_led_blink_eq=True,
        cut_target_ra_gte=85,
        cut_target_ra_lte=82,
        cut_target_dec_gte=23,
        cut_target_dec_lte=21,
        cut_nsb_rate_gte=0.6 ,
        cut_nsb_rate_lte=.1,
        cut_r_gte=None ,
        cut_r_lte=None,
        cut_n_island_gte=2,
        alphas_min=alphas_min,
        plot_scan2d=plot_scan2d,
        plot_showers_center=plot_showers_center,
        plot_hillas=plot_hillas,
        plot_correlation_all=plot_correlation_all,
        plot_correlation_selected=plot_correlation_selected,
        plot_correlation_cut=plot_correlation_cut,
        plot_map_disp=plot_map_disp,
        xis=xis,
        print_events=0,
        disable_bar=disable_bar,
    )


if __name__ == '__main__':
    entry()
