#!/usr/bin/env python
"""
Create and plot a pulse template from the 2D histograms of the pusle shape for
each pixel (output of digicam-pulse-shape).
Usage:
  digicam-pulse-template [options] <input_files>...

Options:
  -h --help                 Show this screen.
  <INPUT>                   List of path to fits files containing 2D histograms
                            to combine to create the pulse template.
  --output=PATH             Path to the pulse template file to be created.
                            It is a text file with 3 columns: time, amplitude
                            and standard deviation.
                            If set to "none" the file is not created.
                            [Default: none]
  --plot=PATH               Path to the output plot. Will show the average
                            over all events of the trigger rate.
                            If set to "show", the plot is displayed and not
                            saved. If set to "none", no plot is done.
                            [Default: show]
"""
import matplotlib.pyplot as plt
from docopt import docopt
import os

from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.utils.docopt import convert_text


def main(input_files, output=None, plot="show"):
    template = NormalizedPulseTemplate.create_from_datafiles(
        input_files=input_files,
        min_entries_ratio=0.1
    )
    if output is not None:
        if os.path.exists(output):
            os.remove(output)
        template.save(output)
    if plot is not None:
        fig, ax = plt.subplots(1, 1)
        template.plot(axes=ax)
        if plot.lower() == "show":
            plt.show()
        else:
            plt.savefig(plot)
        plt.close(fig)


def entry():
    args = docopt(__doc__)
    inputs = args['<input_files>']
    output = convert_text(args['--output'])
    plot = convert_text(args['--plot'])
    main(
        input_files=inputs,
        output=output,
        plot=plot
    )


if __name__ == '__main__':
    entry()
