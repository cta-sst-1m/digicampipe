"""
Show stars position on top of NSB in each camera pixels
Usage:
  follow_stars.py [options] [--] <INPUTS> ...

Options:
  --help                        Show this
  <INPUTS>                      Zfits data files or output fits file containing
                                the baselines (obtained with --output).
  --aux_basepath=DIR            Base directory for the auxilary data.
                                If set to "search", It will try to determine it
                                from the input files.
                                [Default: search]
  --dark_hist=LIST              Histogram of ADC samples during dark run.
                                Output of raw.py on dark data.
  --output=FILE                 Fits file containing the baselines. Set to none
                                to not create that file. No output is created
                                if the inputs are not zfits data file.
                                [Default: none]
  --max_events=N                Maximum number of events to analyze
  --plot=FILE                   path to the output plot. Will show the
                                evolution of the NSB as a video.
                                If set to "show", the video is displayed and
                                not saved.
                                If set to "none", no video is done.
                                [Default: show]
  --norm=TEXT                   Norm to use for the nsb scale. must be
                                "lin" or "log", [Default: log]
  --plot_baselines              enable the plot of the history of various
                                baselines [Default: True]
  --parameters=FILE             Calibration parameters file path
  --template=FILE               Pulse template file path
  --bias_resistance=FLOAT       Bias resistance in Ohm. [Default: 1e4]
  --cell_capacitance=FLOAT      Cell capacitance in Farad. [Default: 5e-14]
"""
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
from ctapipe.visualization import CameraDisplay
from docopt import docopt
import yaml
from astropy import units as u
from astropy.time import Time
from astropy.table import Table
from pkg_resources import resource_filename
from datetime import datetime

from histogram.histogram import Histogram1D
from digicampipe.calib.baseline import _compute_nsb_rate
from digicampipe.instrument.camera import DigiCam
from digicampipe.utils.docopt import convert_text, convert_float, convert_int
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.calib.charge import _get_average_matrix_bad_pixels
from digicampipe.io.event_stream import calibration_event_stream, \
    add_slow_data_calibration
from digicampipe.utils.transformations import transform_azel_to_xy, \
    get_stars_in_fov


def nsb_rate(
        files, aux_basepath, dark_histo_file, param_file, template_filename,
        output=None, plot="show", plot_nsb_range=None, norm="log",
        plot_baselines=False, disable_bar=False, max_events=None, n_skip=10,
        stars=True,
        bias_resistance=1e4 * u.Ohm, cell_capacitance=5e-14 * u.Farad
):
    files = np.atleast_1d(files)

    if len(files) == 1 and not files[0].endswith('.fz'):
        table = Table.read(files[0])[:max_events]
        data = dict(table)
        data['nsb_rate'] = np.array(data['nsb_rate']) * u.GHz
    else:
        dark_histo = Histogram1D.load(dark_histo_file)
        n_pixel = len(DigiCam.geometry.neighbors)
        pixels = np.arange(n_pixel, dtype=int)
        with open(param_file) as file:
            pulse_template = NormalizedPulseTemplate.load(template_filename)
            pulse_area = pulse_template.integral() * u.ns
            charge_to_amplitude = pulse_template.compute_charge_amplitude_ratio(7, 4)
            calibration_parameters = yaml.load(file)
            gain_integral = np.array(calibration_parameters['gain'])
            gain_amplitude = gain_integral * charge_to_amplitude
            crosstalk = np.array(calibration_parameters['mu_xt'])
        events = calibration_event_stream(files, max_events=max_events,
                                          disable_bar=disable_bar)
        events = add_slow_data_calibration(
            events, basepath=aux_basepath,
            aux_services=('DriveSystem', )
        )
        data = {
            "baseline": [],
            "nsb_rate": [],
            "good_pixels_mask": [],
            "timestamp": [],
            "event_id": [],
            "az": [],
            "el": [],
        }
        bad_pixels = get_bad_pixels(
            calib_file=param_file, nsigma_gain=5, nsigma_elecnoise=5,
            dark_histo=dark_histo_file, nsigma_dark=8, plot=None, output=None
        )
        events_skipped = 0
        for event in events:
            if event.event_type.INTERNAL not in event.event_type:
                continue
            events_skipped += 1
            if events_skipped < n_skip:
                continue
            events_skipped = 0
            data['baseline'].append(event.data.digicam_baseline)
            baseline_shift = event.data.digicam_baseline - dark_histo.mean()
            rate = _compute_nsb_rate(
                baseline_shift=baseline_shift, gain=gain_amplitude,
                pulse_area=pulse_area, crosstalk=crosstalk,
                bias_resistance=bias_resistance, cell_capacitance=cell_capacitance
            )
            bad_pixels_event = np.unique(np.hstack(
                (
                    bad_pixels,
                    pixels[rate < 0],
                    pixels[rate > 5 * u.GHz]
                )
            ))
            avg_matrix = _get_average_matrix_bad_pixels(
                DigiCam.geometry, bad_pixels_event
            )
            good_pixels_mask = np.ones(n_pixel, dtype=bool)
            good_pixels_mask[bad_pixels_event] = False
            good_pixels = pixels[good_pixels_mask]
            rate[bad_pixels_event] = avg_matrix[bad_pixels_event, :].dot(
                rate[good_pixels]
            )
            data['good_pixels_mask'].append(good_pixels_mask)
            data['timestamp'].append(event.data.local_time)
            data['event_id'].append(event.event_id)
            data['nsb_rate'].append(rate)
            data['az'].append(event.slow_data.DriveSystem.current_position_az)
            data['el'].append(event.slow_data.DriveSystem.current_position_el)
        data['nsb_rate'] = np.array(data['nsb_rate']) * u.GHz
        if output is not None:
            table = Table(data)
            if os.path.isfile(output):
                os.remove(output)
            table.write(output, format='fits')

    time_obs = Time(
        np.array(data['timestamp'], dtype=np.float64) * 1e-9,
        format='unix'
    )
    if plot_baselines:
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=50)
        baseline_std = np.std(data['baseline'], axis=0)
        ax1.hist(baseline_std, 100)
        ax1.set_xlabel('std(baseline) [LSB]')
        # pixels_shown = np.arange(len(baseline_std))[baseline_std > 10]
        pixels_shown = [834,]
        ax2.plot_date(
            time_obs.to_datetime(),
            data['baseline'][:, pixels_shown],
            '-'
        )
        ax2.set_xlabel('time')
        ax2.set_ylabel('baseline [LSB]')
        plt.tight_layout()
        plt.show()
        plt.close(fig2)

    az_obs = np.array(data['az']) * u.deg
    el_obs = np.array(data['el']) * u.deg
    n_event = len(data['timestamp'])
    fig1, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=50)
    date = datetime.fromtimestamp(data['timestamp'][0]*1e-9)
    date_str = date.strftime("%H:%M:%S")
    display = CameraDisplay(
        DigiCam.geometry, ax=ax, norm=norm,
        title='NSB rate [GHz], t=' + date_str
    )
    rate_ghz = np.array(data['nsb_rate'][0].to(u.GHz).value)
    display.image = rate_ghz
    if plot_nsb_range is None:
        min_range_rate = np.max([np.min(rate_ghz), 50e-3])
        plot_nsb_range = (min_range_rate, np.max(rate_ghz))
    display.set_limits_minmax(*plot_nsb_range)
    display.add_colorbar(ax=ax)
    bad_pixels = np.arange(
        len(data['good_pixels_mask'][0])
    )[~data['good_pixels_mask'][0]]
    display.highlight_pixels(bad_pixels, color='r', linewidth=2)
    display.axes.set_xlim([-500., 500.])
    display.axes.set_ylim([-500., 500.])
    plt.tight_layout()
    if stars is True:
        stars_az, stars_alt, stars_pmag = get_stars_in_fov(
            az_obs[0], el_obs[0], time_obs
        )
        stars_x, stars_y = transform_azel_to_xy(
            stars_az, stars_alt, az_obs, el_obs
        )
        point_stars = []
        for index_star in range(len(stars_pmag)):
            point_star, = ax.plot(
                stars_x[index_star, 0],
                stars_y[index_star, 0],
                'ok',
                ms=20-2*stars_pmag[index_star],
                mew=3,
                mfc='None'
            )
            point_stars.append(point_star)

    def update(i, display):
        print('frame', i, '/', len(data['timestamp']))
        display.image = data['nsb_rate'][i].to(u.GHz).value
        date = datetime.fromtimestamp(data['timestamp'][i] * 1e-9)
        date_str = date.strftime("%H:%M:%S")
        display.axes.set_title('NSB rate [GHz], t=' + date_str)
        bad_pixels = np.arange(
            len(data['good_pixels_mask'][i])
        )[~data['good_pixels_mask'][i]]
        display.highlight_pixels(
            bad_pixels, color='r', linewidth=2
        )
        if stars is True:
            for index_star in range(len(stars_pmag)):
                point_stars[index_star].set_xdata(
                    stars_x[index_star, i]
                )
                point_stars[index_star].set_ydata(
                    stars_y[index_star, i]
                )

    anim = FuncAnimation(
        fig1,
        update,
        frames=len(data['timestamp']),
        interval=20,
        fargs=(display, )
    )
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='Y. Renier'),
                    bitrate=4000, codec='h263p')

    output_path = os.path.dirname(plot)
    if plot == "show" or \
            (output_path != "" and not os.path.isdir(output_path)):
        if not plot == "show":
            print('WARNING: Path ' + output_path + ' for output trigger ' +
                  'uniformity does not exist, displaying the plot instead.\n')
        display.enable_pixel_picker()
        plt.show()
    else:
        anim.save(plot, writer=writer)
        print(plot, 'created')
    plt.close(fig1)


def entry():
    args = docopt(__doc__)
    files = np.atleast_1d(args['<INPUTS>'])
    aux_basepath = convert_text(args['--aux_basepath'])
    if aux_basepath.lower() == "search":
        input_dir = np.unique([os.path.dirname(file) for file in files])
        if len(input_dir) > 1:
            raise AttributeError(
                "Input files must be from the same directory " +
                "when searching for auxiliaries files"
            )
        input_dir = input_dir[0]
        aux_basepath = input_dir.replace('/raw/', '/aux/')
        if not os.path.isdir(aux_basepath):
            aux_basepath = aux_basepath.replace('/SST1M_01', '/SST1M01')
        if not os.path.isdir(aux_basepath):
            raise AttributeError(
                "Searching for auxiliaries files failed. " +
                "Please use --aux_basepath=PATH"
            )
        print('expecting aux files in', aux_basepath)
    dark_histo_file = convert_text(args['--dark_hist'])
    param_file = convert_text(args['--parameters'])
    if param_file is None:
        param_file = resource_filename(
            'digicampipe',
            os.path.join(
                'tests',
                'resources',
                'calibration_20180814.yml'
            )
        )
    template_filename = convert_text(args['--template'])
    if template_filename is None:
        template_filename = resource_filename(
            'digicampipe',
            os.path.join(
                'tests',
                'resources',
                'pulse_template_all_pixels.txt'
            )
        )
    output = convert_text(args['--output'])
    max_events = convert_int(args['--max_events'])
    plot = convert_text(args['--plot'])
    norm = convert_text(args['--norm'])
    plot_baselines = args['--plot_baselines']
    bias_resistance = convert_float(args['--bias_resistance']) * u.Ohm
    cell_capacitance = convert_float(args['--cell_capacitance']) * u.Farad
    nsb_rate(
        files, aux_basepath, dark_histo_file, param_file, template_filename,
        output=output, norm=norm, plot_baselines=plot_baselines,
        plot=plot, bias_resistance=bias_resistance, max_events=max_events,
        stars=True, cell_capacitance=cell_capacitance
    )


if __name__ == '__main__':
    entry()