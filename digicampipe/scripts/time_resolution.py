#!/usr/bin/env python
"""
measure the time offsets using template fitting. Create a file
time_acXXX_dcYYY.npz for eac AC and DC level with XXX and YYY the value of resp.
the AC and DC level.
Usage:
  digicam-time-resolution [options] [--] <INPUT>...

Options:
  -h --help                     Show this screen.
  --ac_levels=LIST              Comma separated list of AC levels for each
                                input file. Must be of the same size as the
                                number of input file.
  --dc_levels=LIST              Comma separated list of DC levels for each
                                input file. Must be a single value or the same
                                size as the number of input files. [default: 0]
  --max_events=INT              Maximum number of events to analyze. Set to
                                none to use all events in the input files.
                                [default: none]
  --delay_step_ns=FLOAT         Unit of delay in ns used to shift the template
                                [default: 0.1]
  --time_range_ns=<RANGE>       Coma separated interval in ns for the pulse
                                template.
                                [default: -9.,39.]
  --normalize_range=<RANGE>     index of the border samples around the max used
                                for normalization. If set to 0,0 the
                                normalization is by the amplitude of the max.
                                [default: -3,4]
  --parameters=FILE             Calibration parameters file path. If set to
                                none, the default one is used.
                                [default: none]
  --template=FILE               Pulse template file path. If set to none, the
                                default one is used.
                                [default: none]
  --output=PATH                 Path to a directory where results will be
                                stored.
                                [default: ./]
"""
from docopt import docopt
from pkg_resources import resource_filename
from glob import glob
import os
import numpy as np
import yaml
from matplotlib import pyplot as plt
from ctapipe.visualization import CameraDisplay

from digicampipe.utils.docopt import convert_int, convert_float, convert_text,\
    convert_list_float, convert_list_int
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.baseline import fill_digicam_baseline, subtract_baseline
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.instrument.camera import DigiCam
from digicampipe.instrument.light_source import ACLED


parameters_default = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)
template_default = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_template_all_pixels.txt'
    )
)

cone_pde = 0.88
sipm_pde = 0.35
window_pde = 0.9
pde = cone_pde * sipm_pde * window_pde


def plot_rms_difference(data_file, figure_file, n_pe=20*pde, vmin=0., vmax=2.):
    file_calib = os.path.join('mpe_fit_results_combined.npz')
    data_calib = np.load(file_calib)
    ac_led = ACLED(
        data_calib['ac_levels'][:, 0],
        data_calib['mu'],
        data_calib['mu_error']
    )

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

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    rms_diff = np.sqrt(std_t_npe[:, None]**2 + std_t_npe[None, :]**2)
    h = ax.pcolormesh(rms_diff, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(h, ax=ax)
    cb.ax.set_ylabel('rms time difference [ns]')
    ax.set_xlabel('pixel')
    ax.set_ylabel('pixel')
    plt.title('125 MHz NSB, rms time difference at {:.1f} p.e.'.format(n_pe))
    plt.tight_layout()
    plt.savefig(figure_file)
    print(figure_file, 'created')


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
    ax.xaxis.set_label_position('top')
    ax.grid(True)
    ax.legend()
    ax2 = ax.twiny()  # instantiate a second axes that shares the same y-axis
    ax2.plot(1e-5, 1e-5, alpha=0)
    ax2.tick_params(axis='x')
    ax2.set_xlim(x_min / pde, x_max / pde)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xscale(xscale)
    ax2.set_yscale(yscale)
    ax.xaxis.tick_top()
    ax2.xaxis.tick_bottom()
    ax2.set_xlabel('$N_\gamma$ [ph.]')
    ax2.xaxis.set_label_position('bottom')


def plot_resol(data_file, figure_file, legend):
    file_calib = os.path.join('mpe_fit_results_combined.npz')
    data_calib = np.load(file_calib)
    ac_led = ACLED(
        data_calib['ac_levels'][:, 0],
        data_calib['mu'],
        data_calib['mu_error']
    )

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

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_zone(
        true_pe,
        std_t_all,
        [np.logspace(.5, 2.75, 101), np.logspace(-1.3, 0.5, 101)],
        ax,
        legend,
        yscale='log'
    )
    ax.set_ylabel('time resolution [ns]')

    plt.tight_layout()
    plt.savefig(figure_file)


def plot_all(data_file, figure_file='time_resolution.png'):
    file_calib = os.path.join('mpe_fit_results_combined.npz')
    data_calib = np.load(file_calib)
    ac_led = ACLED(
        data_calib['ac_levels'][:, 0],
        data_calib['mu'],
        data_calib['mu_error']
    )

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

    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
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
    plt.savefig(figure_file)


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
        dc_levels.append( data['dc_level'])
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


def analyse_ACDC_level(
    files, max_events, delay_step_ns, time_range_ns, sampling_ns,
    normalize_range, parameters, template, adc_noise
):
    with open(parameters) as parameters_file:
        calibration_parameters = yaml.load(parameters_file)
    gain_pixels = np.array(calibration_parameters['gain'])
    normalize_slice = np.arange(normalize_range[0], normalize_range[1]+1,
                                dtype=int)
    sample_template = np.arange(time_range_ns[0], time_range_ns[1], sampling_ns)
    n_sample_template = len(sample_template)
    template = NormalizedPulseTemplate.load(template)
    delays = np.arange(-4, 4, delay_step_ns)
    n_delays = len(delays)
    templates_ampl = np.zeros([n_delays, n_sample_template])
    templates_std = np.zeros([n_delays, n_sample_template])
    index_max_template = np.zeros(n_delays, dtype=int)
    for i, delay in enumerate(delays):
        ampl_templ = template(sample_template + delay)
        index_max_template[i] = np.argmax(ampl_templ)
        range_integ = index_max_template[i] + normalize_slice
        norm_templ = np.sum(ampl_templ[range_integ])
        templates_ampl[i, :] = ampl_templ / norm_templ
        templates_std[i, :] = template.std(sample_template + delay) / norm_templ
    max_ampl_one_pe = np.max(templates_ampl)
    events = calibration_event_stream(files, max_events=max_events,
                                      disable_bar=True)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    rows_norm = np.tile(
        np.arange(1296, dtype=int)[:, None],
        [1, len(normalize_slice)]
    )
    samples_events = None
    t_fit = []
    charge = []
    for event in events:
        if samples_events is None:
            n_sample = event.data.adc_samples.shape[1]
            samples_events = np.arange(n_sample) * sampling_ns
        adc_samples = event.data.adc_samples
        idx_sample_max = np.argmax(adc_samples, axis=1)
        column_norm = normalize_slice[None, :] + idx_sample_max[:, None]
        # we skip pixels with max too close to the limit of the sampling window
        # to be sure to be able to integrate and get normalization
        good_pix = np.logical_and(
            np.all(column_norm < n_sample, axis=1),
            np.all(column_norm >= 0, axis=1)
        )
        # we skip pixels with max too close to the limit of the sampling window
        # to be sure to be able to compare with full template
        mean_index_max_template = int(np.round(np.mean(index_max_template)))
        index_template_rel = idx_sample_max - mean_index_max_template
        good_pix = np.logical_and(
            good_pix,
            index_template_rel >= 0
        )
        good_pix = np.logical_and(
            good_pix,
            index_template_rel + n_sample_template - 1 < n_sample
        )
        # we discard pixels with less that few pe
        good_pix = np.logical_and(
            good_pix,
            np.max(adc_samples, axis=1) > 3.5 / max_ampl_one_pe
        )
        # discard pixels with max pulse not around the right position
        good_pix = np.logical_and(
            good_pix,
            idx_sample_max >= 15
        )
        good_pix = np.logical_and(
            good_pix,
            idx_sample_max <= 16
        )
        sample_norm = adc_samples[rows_norm[good_pix, :], column_norm[good_pix, :]]
        norm_pixels = np.sum(sample_norm, axis=1)
        #discard pixels where charge is <= 2.5 LSB (0.5 pe), as normalization
        # is then meaningless
        norm_all = np.zeros(1296)
        norm_all[good_pix] = norm_pixels
        good_pix = np.logical_and(
            good_pix,
            norm_all > 2.5
        )
        norm_pixels = norm_pixels[norm_pixels > 2.5]

        charge_good_pix = norm_pixels / gain_pixels[good_pix]
        adc_samples_norm = adc_samples[good_pix, :] / norm_pixels[:, None]
        n_good_pix = int(np.sum(good_pix))

        column_chi2 = index_template_rel[good_pix, None, None] + np.arange(n_sample_template, dtype=int)[None, None, :]
        row_chi2 = np.tile(np.arange(n_good_pix)[:, None, None], [1, n_delays, n_sample_template])
        adc_samples_compared = adc_samples_norm[row_chi2, column_chi2]
        residual = adc_samples_compared - templates_ampl[None, :, :]
        error_squared = templates_std[None, :, :] ** 2 \
            + (adc_noise/norm_pixels[:, None, None]) ** 2
        chi2 = np.sum(residual**2/error_squared, axis=2)/(n_sample_template - 1)

        t_fit_all = np.ones(1296) * np.nan
        # estimate offset from min chi2
        idx_delay_min = np.argmin(chi2, axis=1)
        delays_min = delays[idx_delay_min]
        delays_min[chi2[np.arange(n_good_pix), idx_delay_min] > 20] = np.nan
        t_fit_all[good_pix] = delays_min
        """
        # estimate offset from parabola fit of chi2 = f(t)
        idx_min = np.argmin(chi2, axis=1)
        idx_min[idx_min < 3] = 3
        idx_min[idx_min + 4 > n_delays] = n_delays - 4
        indexes_fits = np.arange(-3, 4, dtype=int)[None, :] + idx_min[: , None]
        rows_chi2_fit = np.tile(np.arange(n_good_pix)[:, None], [1, 7])
        p = np.polyfit(
            np.arange(-3, 4) * sampling_ns,
            chi2[rows_chi2_fit, indexes_fits].T,
            2
        )
        relative_delays_fit = -p[1] / (2 * p[0])
        delays_fit = relative_delays_fit + delays[idx_min]
        t_fit_all[good_pix] = delays_fit
        """

        t_fit.append(-t_fit_all + idx_sample_max * sampling_ns)
        charge_all = np.ones(1296) * np.nan
        charge_all[good_pix] = charge_good_pix
        charge.append(charge_all)
    t_fit = np.array(t_fit)
    charge = np.array(charge)
    return charge, t_fit


def main(files, ac_levels, dc_levels, max_events, delay_step_ns, time_range_ns,
         sampling_ns, normalize_range, parameters, template, adc_noise, output):
    unique_ac_dc, inverse = np.unique(
        np.vstack([ac_levels, dc_levels]).T,
        axis=0,
        return_inverse=True
    )
    files = np.array(files)
    for i, (ac_level, dc_level) in enumerate(unique_ac_dc):
        files_level = files[inverse == i]
        print('analyze file with AC DAC =', ac_level, 'DC DAC =', dc_level)
        charge, t_fit = analyse_ACDC_level(
            files_level, max_events, delay_step_ns, time_range_ns, sampling_ns,
            normalize_range, parameters, template, adc_noise
        )
        filename = os.path.join(
            output,
            'time_ac{}_dc{}.npz'.format(ac_level, dc_level)
        )
        mean_charge = np.nanmean(charge, axis=0)
        std_charge = np.nanstd(charge, axis=0)
        mean_t = np.nanmean(t_fit, axis=0)
        std_t = np.nanstd(t_fit, axis=0)
        np.savez(
            filename,
            mean_charge=mean_charge,
            std_charge=std_charge,
            mean_t=mean_t,
            std_t=std_t,
            ac_level=ac_level,
            dc_level=dc_level
        )


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    ac_levels = convert_list_int(args['--ac_levels'])
    dc_levels = convert_list_int(args['--dc_levels'])
    if len(dc_levels) == 1:
        dc_levels = [dc_levels[0],] * len(ac_levels)
    assert len(ac_levels) == len(files)
    max_events = convert_int(args['--max_events'])
    delay_step_ns = convert_float(args['--delay_step_ns'])
    time_range_ns = convert_list_float(args['--time_range_ns'])
    normalize_range = convert_list_int(args['--normalize_range'])
    sampling_ns = 4
    parameters = convert_text(args['--parameters'])
    template = convert_text(args['--template'])
    output = convert_text(args['--output'])
    if parameters is None:
        parameters = parameters_default
    if template is None:
        template = template_default
    main(
        files=files,
        ac_levels=ac_levels,
        dc_levels=dc_levels,
        max_events=max_events,
        delay_step_ns=delay_step_ns,
        time_range_ns=time_range_ns,
        sampling_ns=sampling_ns,
        normalize_range=normalize_range,
        parameters=parameters,
        template=template,
        adc_noise=1.,
        output=output
    )


if __name__ == '__main__':
    entry()
    """
    test_files = glob('./time_ac*_dc0.npz')
    data_combined = 'time_resolution_test.npz'
    combine(
        test_files,
        data_combined
    )
    plot_all(data_combined, figure_file='time_analysis_test.png')

    for dc in range(200, 330, 10):
        timing_level_files = glob(
            os.path.join(
                '/mnt/baobab/sst1m/analyzed/timing_resolution/20180628/',
                'time_ac*_dc{}.npz'.format(dc)
            )
        )
        data_combined = 'time_resolution_dc{}.npz'.format(dc)
        combine(
            timing_level_files,
            data_combined
        )
        #data_combined = 'time_resolution_dc{}.npz'.format(dc)
        figure_file = 'time_analysis_dc{}.png'.format(dc)
        plot_all(data_combined, figure_file=figure_file)
        print(figure_file, 'created with', len(timing_level_files), 'AC lvl')
    plot_resol(
        'time_resolution_dc290.npz',
        figure_file='time_resolution_dc290.png',
        legend='125MHz NSB, camera average'
    )
    plot_rms_difference(
        'time_resolution_dc290.npz',
        figure_file='rms_difference_dc290.png',
        n_pe=1.5,
        vmax=3
    )
    """
