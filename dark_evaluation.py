#!/usr/bin/env python
'''
Extract dark.npz and bias_curve_dark.npz from observations.

    Example call:
    dark_evaluation.py . path/to/SST1M01_20171030.002.fits.fz

Usage:
  dark_evaluation.py [-o <dir>] [-i <file>]...

Options:
  -h --help     Show this screen.
  -i <file>, --input=<file>  input file [default: {example_file}]
  -o <dir>, --outdir=<dir>  output directory [default: .]

'''
from digicampipe.calib.camera import filter, r0
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_adc import save_dark
from digicampipe.io.save_bias_curve import save_bias_curve
from cts_core.camera import Camera
from digicampipe.utils import geometry
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pkg_resources
from os import path
from docopt import docopt

example_file_path = pkg_resources.resource_filename(
        'digicampipe',
        path.join(
            'tests',
            'resources',
            'example_100_evts.000.fits.fz'
        )
    )
__doc__ = __doc__.format(
    example_file=example_file_path
    )


def main(output_directory, files):
    camera_config_file = pkg_resources.resource_filename(
        'digicampipe',
        path.join(
            'tests',
            'resources',
            'camera_config.cfg'
        )
    )

    dark_file_path = path.join(output_directory, 'dark.npz')
    dark_trigger_file_path = path.join(output_directory, 'bias_curve_dark.npz')

    thresholds = np.arange(0, 400, 10)
    unwanted_patch = [306, 318, 330, 342, 200]
    unwanted_cluster = [200]
    blinding = True

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Define the event stream
    data_stream = event_stream(
        file_list=files,
        expert_mode=True,
        camera_geometry=digicam_geometry,
        camera=digicam
    )
    data_stream = filter.set_patches_to_zero(
        data_stream,
        unwanted_patch=unwanted_patch)
    data_stream = r0.fill_trigger_input_7(data_stream)
    # Fill the flags (to be replaced by Digicam)
    data_stream = filter.filter_event_types(data_stream, flags=[8])

    data_stream = save_bias_curve(
        data_stream,
        thresholds=thresholds,
        blinding=blinding,
        output_filename=dark_trigger_file_path,
        unwanted_cluster=unwanted_cluster
    )

    data_stream = save_dark(data_stream, dark_file_path)

    for _ in tqdm(data_stream):
        pass

    # Filter the events for display
    data_dark = np.load(dark_file_path)
    data_rate = np.load(dark_trigger_file_path)

    plt.figure()
    plt.hist(data_dark['baseline'], bins='auto')
    plt.xlabel('dark baseline [LSB]')
    plt.ylabel('count')

    plt.figure()
    plt.hist(data_dark['standard_deviation'], bins='auto')
    plt.xlabel('dark std [LSB]')
    plt.ylabel('count')

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(
        x=data_rate['threshold'],
        y=data_rate['rate'] * 1E9,
        yerr=data_rate['rate_error'] * 1E9,
        label='Blinding : {}'.format(blinding)
    )
    axis.set_ylabel('rate [Hz]')
    axis.set_xlabel('threshold [LSB]')
    axis.set_yscale('log')
    axis.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)
    main(
        output_directory=args['<output_directory>'],
        files=args['<files>']
    )
