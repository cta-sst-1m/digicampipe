#!/usr/bin/env python
"""
Create and plot a pulse template from the 2D histograms of the pulse shape for
each pixel (output of digicam-pulse-shape).
Usage:
  digicam-pulse-template [options] <input_files>...

Options:
  -h --help                 Show this screen.
  <input_files>             List of path to fits files containing 2D histograms
                            to combine to create the pulse template.
  --pixels=LIST             List of pixels to use to create the template. If
                            "none", all pixels are used. [Default: none]
  --output=PATH             Path to the pulse template file to be created.
                            It is a text file with 3 columns: time, amplitude
                            and standard deviation.
                            If set to "none" the file is not created.
                            [Default: none]
  --plot=PATH               Path to the output plot. Will plot the normalized
                            pulse amplitude function of time.
                            If set to "show", the plot is displayed and not
                            saved. If set to "none", no plot is done.
                            [Default: show]
  --plot_separated=PATH     Path to the output plot. Will plot the normalized
                            pulse amplitude function of time for each of the
                            input files.
                            If set to "show", the plot is displayed and not
                            saved. If set to "none", no plot is done.
                            [Default: none]
  --xscale=STRING           "linear", "log", "symlog" or "logit". The X axis
                            scale type to apply. See Axes.set_xscale().
                            [Default: linear]
  --yscale=STRING           "linear", "log", "symlog" or "logit". The Y axis
                            scale type to apply. See Axes.set_yscale().
                            [Default: linear]
"""
import matplotlib.pyplot as plt
from docopt import docopt
import os

from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.utils.docopt import convert_text, convert_pixel_args
from digicampipe.visualization.plot import plot_pulse_templates


def main(input_files, output=None, plot="show", plot_separated=None,
         pixels=None, xscale="linear", yscale="linear"):
    if output is not None or plot is not None:
        template = NormalizedPulseTemplate.create_from_datafiles(
            input_files=input_files,
            min_entries_ratio=0.1,
            pixels=pixels
        )
        if output is not None:
            if os.path.exists(output):
                os.remove(output)
            template.save(output)
            print(output, 'created')
        if plot is not None:
            fig, ax = plt.subplots(1, 1)
            template.plot(axes=ax)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if plot.lower() == "show":
                plt.show()
            else:
                plt.savefig(plot)
                print(plot, 'created')
            plt.close(fig)
    if plot_separated is not None:
        fig, ax = plt.subplots(1, 1)
        plot_pulse_templates(
            input_files, xscale=xscale, yscale=yscale, axes=ax
        )
        if plot_separated.lower() == "show":
            plt.show()
        else:
            plt.savefig(plot_separated)
            print(plot_separated, 'created')
        plt.close(fig)


def entry():
    args = docopt(__doc__)
    inputs = args['<input_files>']
    output = convert_text(args['--output'])
    plot = convert_text(args['--plot'])
    plot_separated = convert_text(args['--plot_separated'])
    pixels = convert_pixel_args(args['--pixels'])
    xscale = args['--xscale']
    yscale = args['--yscale']
    main(
        input_files=inputs,
        output=output,
        plot=plot,
        plot_separated=plot_separated,
        xscale=xscale,
        yscale=yscale,
        pixels=pixels
    )


if __name__ == '__main__':
    entry()
