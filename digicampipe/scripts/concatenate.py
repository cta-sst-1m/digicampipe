#!/usr/bin/env python
"""
concatenate the input fits files to one output.
Useful to merge the output of several runs (f.e. hillas.fits from pipeline.py)

Usage:
  digicam-concatenate [options] <OUTPUT> <INPUTS>...

Options:
  -h --help                   Show this screen.
  --hdu_list=LIST             List of HDU numbers to combine
                              [Default: 1]
"""

from docopt import docopt
from glob import glob
from digicampipe.utils.docopt import convert_list_int
import re
import fitsio
import shutil


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def concatenate(inputs, output, hdus=[1]):

    if len(inputs) < 2:
        raise AttributeError('digicam-concatenate must take 1 output and at '
                             'least 2 inputs files as arguments')

    shutil.copyfile(inputs[0], output)

    with fitsio.FITS(output, 'rw') as f_out:

        for hdu in hdus:

            for i, input in enumerate(inputs[1:]):

                with fitsio.FITS(input, 'r') as f_in:

                    f_out[hdu].append(f_in[hdu].read())


def entry():

    args = docopt(__doc__)
    inputs = args['<INPUTS>']
    if len(inputs) == 1:
        inputs = glob(inputs[0])
        inputs.sort(key=alphanum_key)
    output = args['<OUTPUT>']
    hdus = convert_list_int(args['--hdu_list'])

    concatenate(inputs, output, hdus=hdus)


if __name__ == '__main__':

    entry()
