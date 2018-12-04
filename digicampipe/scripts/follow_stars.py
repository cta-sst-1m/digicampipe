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
  --plot=FILE                   path to the output plot. Will show the average
                                over all events of the NSB.
                                If set to "show", the plot is displayed and not
                                saved.
                                If set to "none", no plot is done.
                                [Default: show]
  --parameters=FILE             Calibration parameters file path
  --template=FILE               Pulse template file path
  --bias_resistance=FLOAT       Bias resistance in Ohm. [Default: 1e4]
  --cell_capacitance=FLOAT      Cell capacitance in Farad. [Default: 5e-14]
"""
import os
# import matplotlib
# matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
from ctapipe.visualization import CameraDisplay
from docopt import docopt
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.table import Table
from pkg_resources import resource_filename

from histogram.histogram import Histogram1D
from digicampipe.calib.baseline import _compute_nsb_rate
from digicampipe.instrument.camera import DigiCam
from digicampipe.utils.docopt import convert_text, convert_float, convert_int
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.calib.charge import _get_average_matrix_bad_pixels
from digicampipe.io.event_stream import calibration_event_stream, \
    add_slow_data_calibration


def nsb_rate(
        files, aux_basepath, dark_histo_file, param_file, template_filename,
        output=None,
        plot="show", plot_nsb_range=None, norm="log", disable_bar=False,
        max_events=None, stars=('Capella',), mm_per_deg=-100.,
        site=(50.049683 * u.deg, 19.944544 * u.deg, 209 * u.m),  # site_location
        bias_resistance=1e4 * u.Ohm, cell_capacitance=5e-14 * u.Farad
):
    files = np.atleast_1d(files)
    site_location = EarthLocation(lat=site[0], lon=site[1], height=site[2])
    if len(files) == 1 and not files[0].endswith('.fz'):
        table = Table.read(files[0])
        data = dict(table)
        data['nsb_rate'] = data['nsb_rate'] * u.GHz
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
            #  'DigicamSlowControl', 'MasterSST1M', 'SafetyPLC', #
            #  'PDPSlowControl')
        )
        data = {
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
        event_counter = 0
        for event in events:
            if event.event_type.INTERNAL not in event.event_type:
                continue
            event_counter += 1
            if event_counter < 100:
                continue
            event_counter = 0
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
        if output is not None:
            table = Table(data)
            if os.path.isfile(output):
                os.remove(output)
            table.write(output, format='fits')
    az_obs = np.array(data['az'])
    el_obs = np.array(data['el'])
    n_event = len(data['timestamp'])
    stars_x = np.zeros([len(stars), n_event])
    stars_y = np.zeros([len(stars), n_event])
    for star_idx, star in enumerate(stars):
        skycoord = SkyCoord.from_name(star)
        star_pos = skycoord.transform_to(
            AltAz(
                obstime=Time(
                    np.array(data['timestamp'], dtype=np.float64) * 1e-9,
                    format='unix'
                ),
                location=site_location
            )
        )
        stars_x[star_idx, :] = (np.array(star_pos.az) - az_obs) * mm_per_deg
        stars_y[star_idx, :] = (np.array(star_pos.alt) - el_obs) * mm_per_deg
    fig1, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=50)
    display = CameraDisplay(
        DigiCam.geometry, ax=ax, norm=norm,
        title='NSB rate [GHz], t={}'.format(int(data['timestamp'][0]*1e-9))
    )
    rate_ghz = np.array(data['nsb_rate'][0].to(u.GHz).value)
    display.image = rate_ghz
    if plot_nsb_range is None:
        plot_nsb_range = (np.min(rate_ghz), np.max(rate_ghz))
    display.set_limits_minmax(*plot_nsb_range)
    display.add_colorbar(ax=ax)
    bad_pixels = np.arange(
        len(data['good_pixels_mask'][0])
    )[data['good_pixels_mask'][0]]
    display.highlight_pixels(bad_pixels, color='r', linewidth=2)
    plt.tight_layout()
    points, = ax.plot(stars_y[:, 0], stars_y[:, 0], 'r+', ms=20)

    def update(i, display):
        print('frame', i, '/', len(data['timestamp']))
        display.image = data['nsb_rate'][i].to(u.GHz).value
        t = int(data['timestamp'][i]*1e-9)
        display.axes.set_title('NSB rate [GHz], t={}'.format(t))
        bad_pixels = np.arange(
            len(data['good_pixels_mask'][i])
        )[data['good_pixels_mask'][i]]
        display.highlight_pixels(
            bad_pixels, color='r', linewidth=2
        )
        points.set_xdata(stars_x[:, i])
        points.set_ydata(stars_y[:, i])

    anim = FuncAnimation(
        fig1,
        update,
        frames=len(data['timestamp']),
        interval=50,
        fargs=(display, )
    )
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='y. renier'),
                    bitrate=1800, codec='h263p')

    output_path = os.path.dirname(plot)
    if plot == "show" or \
            (output_path != "" and not os.path.isdir(output_path)):
        if not plot == "show":
            print('WARNING: Path ' + output_path + ' for output trigger ' +
                  'uniformity does not exist, displaying the plot instead.\n')
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
    bias_resistance = convert_float(args['--bias_resistance']) * u.Ohm
    cell_capacitance = convert_float(args['--cell_capacitance']) * u.Farad
    nsb_rate(
        files, aux_basepath, dark_histo_file, param_file, template_filename,
        output=output,
        plot=plot, bias_resistance=bias_resistance, max_events=max_events,
        stars=('Capella',),
        cell_capacitance=cell_capacitance
    )


if __name__ == '__main__':
    entry()