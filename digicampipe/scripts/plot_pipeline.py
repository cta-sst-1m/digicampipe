#!/usr/bin/env python
"""
Plot the output of the pipeline.

Usage:
  digicam-plot-pipeline [options] [--] <INPUT>

Options:
  -h --help                     Show this screen.
  <INPUT>                       Output file from digicam-pipeline.
                                [Default: ./hillas.fits]
  --plot_scan2d=PATH            path to the plot for a 2d scan of the source
                                position for the number of shower with alpha <
                                --alpha_min. If set to "none", the plot is not
                                produced. If set to "show" the plot is
                                displayed instead. [default: none]
  --alpha_min=FLOAT             Minimum alpha angle in degrees that an event
                                must have during the 2D scan to be included.
                                [Default: 5]
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
                                Hillas parameter for event not passing the cuts.
                                If set to "none", the plot
                                is not produced. If set to "show" the plot is
                                displayed instead. [default:none]
"""
import numpy as np
import pandas as pd
from astropy.table import Table
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from digicampipe.utils.docopt import convert_int, convert_list_int, \
    convert_text, convert_float
from digicampipe.image.hillas import correct_alpha_3


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


def hillas_plot(pipeline_data, selection, plot="show"):
    fig = plt.figure(figsize=(15, 15))
    subplot = 0
    for key, val in pipeline_data.items():
        if key in ['border', 'intensity', 'kurtosis', 'event_id',
                   'event_type', 'miss', 'burst', 'saturated']:
            continue
        subplot += 1
        print(subplot, '/', 9, 'plotting', key)
        plt.subplot(3, 3, subplot)
        val_split = [
            val[(~pipeline_data['burst']) & selection],
            val[(~pipeline_data['burst']) & (~selection)]
        ]
        plt.hist(val_split, bins='auto', stacked=True)
        plt.xlabel(key)
        if subplot == 1:
            plt.legend(['2 < l/w < 10', 'l/w cut'])
    plt.tight_layout()
    if plot == "show":
        plt.show()
    else:
        plt.savefig(plot)
        print(plot, 'created')
    plt.close(fig)


def scan_2d_plot(
        pipeline_data, alpha_min=5., plot="show",
        num_steps=200,
        fov=((-500, 500), (-500, 500)),
):
    """
    2D scan of spike in alpha
    :param pipeline_data: hillas parameters data file, output of pipeline.py
    :param alpha_min: events are taken into account as possibly coming from
    the scanned point if the alpha parameter calculated at that point is below
    alpha_min.
    :param plot: path to the plot for a 2d scan of the source position.
    If set to "none", the plot is not produced. If set to "show" the plot
    is displayed instead.
    :param num_steps: number of binning in the FoV
    :param fov: x and y range of the field of view. Format:
    ((x_min, x_max), (y_min, y_max))
    :return: None
    """
    x_fov_start = -1000  # limits of the FoV in mm
    y_fov_start = -1000  # limits of the FoV in mm
    x_fov_end = 1000  # limits of the FoV in mm
    y_fov_end = 1000  # limits of the FoV in mm
    x_fov = np.linspace(fov[0][0], fov[0][1], num_steps)
    y_fov = np.linspace(fov[1][0], fov[1][1], num_steps)
    dx = x_fov[1] - x_fov[0]
    dy = y_fov[1] - y_fov[0]
    x_fov_bins = np.linspace(x_fov_start - dx / 2, x_fov_end + dx / 2,
                             num_steps + 1)
    y_fov_bins = np.linspace(y_fov_start - dy / 2, y_fov_end + dy / 2,
                             num_steps + 1)
    N = np.zeros([num_steps, num_steps], dtype=int)
    i = 0
    print('2D scan calculation:')
    for xi, x in enumerate(x_fov):
        print(round(i / len(x_fov) * 100, 2), '/', 100)  # progress
        for yi, y in enumerate(y_fov):
            data_at_xy = correct_alpha_3(pipeline_data, source_x=x, source_y=y)
            N[yi, xi] = np.sum(data_at_xy['alpha'] < alpha_min)
        i += 1
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x_fov_bins, y_fov_bins, N,
                         rasterized=True, cmap='nipy_spectral')
    plt.ylabel('FOV Y [mm]')
    plt.xlabel('FOV X [mm]')
    cbar = fig.colorbar(pcm)
    cbar.set_label('N of events')
    plt.tight_layout()
    if plot == "show":
        plt.show()
    else:
        plt.savefig(plot)
        print(plot, 'created')
    plt.close(fig)


def cut_data(
        pipeline_data,
        cut_length_gte=None,
        cut_length_lte=None,
        cut_width_gte=None,
        cut_width_lte=None,
        cut_length_over_width_gte=None,
        cut_length_over_width_lte=None,
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None
):
    selection = np.isfinite(pipeline_data['intensity'])
    if cut_length_gte is not None:
        event_pass = pipeline_data['length'] < cut_length_gte
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: length < ', cut_length_gte)
        selection = np.logical_and(selection, event_pass)
    if cut_length_lte is not None:
        event_pass = pipeline_data['length'] > cut_length_lte
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: length > ', cut_length_lte)
        selection = np.logical_and(selection, event_pass)
    if cut_width_gte is not None:
        event_pass = pipeline_data['width'] < cut_width_gte
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: width < ', cut_width_gte)
        selection = np.logical_and(selection, event_pass)
    if cut_width_lte is not None:
        event_pass = pipeline_data['width'] > cut_width_lte
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: width > ', cut_width_lte)
        selection = np.logical_and(selection, event_pass)
    if cut_length_over_width_gte is not None:
        length_over_width = pipeline_data['length'] / pipeline_data['width']
        event_pass = length_over_width < cut_length_over_width_gte
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: l/w < ', cut_length_over_width_gte)
        selection = np.logical_and(selection, event_pass)
    if cut_length_over_width_lte is not None:
        length_over_width = pipeline_data['length'] / pipeline_data['width']
        event_pass = length_over_width > cut_length_over_width_lte
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: l/w > ',
              cut_length_over_width_lte)
        selection = np.logical_and(selection, event_pass)
    if cut_border_eq is not None:
        event_pass = pipeline_data['border'] != cut_border_eq
        selection = np.logical_and(selection, event_pass)
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: border !=', cut_border_eq)
    if cut_burst_eq:
        event_pass = pipeline_data['burst'] != cut_burst_eq
        selection = np.logical_and(selection, event_pass)
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: burst !=', cut_burst_eq)
    if cut_saturated_eq:
        event_pass = pipeline_data['saturated'] != cut_saturated_eq
        selection = np.logical_and(selection, event_pass)
        print(np.sum(event_pass), '/', np.sum(selection),
              'events cut with selection: saturated !=', cut_saturated_eq)
    return selection


def get_data_and_selection(
        hillas_file,
        cut_length_gte=None,
        cut_length_lte=None,
        cut_width_gte=None,
        cut_width_lte=None,
        cut_length_over_width_gte=None,
        cut_length_over_width_lte=None,
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None,
):
    data = Table.read(hillas_file, format='fits')
    data = data.to_pandas()
    data['local_time'] = pd.to_datetime(data['local_time'])
    data = data.set_index('local_time')
    data = data.dropna()

    selection = cut_data(
        pipeline_data=data,
        cut_length_gte=cut_length_gte,
        cut_length_lte=cut_length_lte,
        cut_width_gte=cut_width_gte,
        cut_width_lte=cut_width_lte,
        cut_length_over_width_gte=cut_length_over_width_gte,
        cut_length_over_width_lte=cut_length_over_width_lte,
        cut_border_eq=cut_border_eq,
        cut_burst_eq=cut_burst_eq,
        cut_saturated_eq=cut_saturated_eq
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
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None,
        alpha_min=5.,
        plot_scan2d=None,
        plot_showers_center=None,
        plot_hillas=None,
        plot_correlation_all=None,
        plot_correlation_selected=None,
        plot_correlation_cut=None,
):
    data, selection = get_data_and_selection(
        hillas_file=hillas_file,
        cut_length_gte=cut_length_gte,
        cut_length_lte=cut_length_lte,
        cut_width_gte=cut_width_gte,
        cut_width_lte=cut_width_lte,
        cut_length_over_width_gte=cut_length_over_width_gte,
        cut_length_over_width_lte=cut_length_over_width_lte,
        cut_border_eq=cut_border_eq,
        cut_burst_eq=cut_burst_eq,
        cut_saturated_eq=cut_saturated_eq,
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
        scan_2d_plot(pipeline_data=data[selection_no_burst],
                     alpha_min=alpha_min, plot=plot_scan2d)


def entry():
    args = docopt(__doc__)
    hillas_file = args['<INPUT>']
    alpha_min = convert_float(args['--alpha_min'])
    plot_scan2d = convert_text(args['--plot_scan2d'])
    plot_showers_center = convert_text(args['--plot_showers_center'])
    plot_hillas = convert_text(args['--plot_hillas'])
    plot_correlation_all=convert_text(args['--plot_correl_all'])
    plot_correlation_selected=convert_text(args['--plot_correl_selected'])
    plot_correlation_cut=convert_text(args['--plot_correl_cut'])
    plot_pipeline(
        hillas_file=hillas_file,
        cut_length_gte=None,
        cut_length_lte=25,
        cut_width_gte=None,
        cut_width_lte=15,
        cut_length_over_width_gte=10,
        cut_length_over_width_lte=2,
        cut_border_eq=None,
        cut_burst_eq=None,
        cut_saturated_eq=None,
        alpha_min=alpha_min,
        plot_scan2d=plot_scan2d,
        plot_showers_center=plot_showers_center,
        plot_hillas=plot_hillas,
        plot_correlation_all=plot_correlation_all,
        plot_correlation_selected=plot_correlation_selected,
        plot_correlation_cut=plot_correlation_cut,
    )


if __name__ == '__main__':
    entry()
