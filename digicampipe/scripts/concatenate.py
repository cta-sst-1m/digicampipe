#!/usr/bin/env python
"""
concatenate the input fits files to one output.
Useful to merge the output of several runs (f.e. hillas.fits from pipeline.py)

Usage:
  digicam-concatenate <OUTPUT> <INPUTS>...

Options:
  -h --help                   Show this screen.
"""

from astropy.table import Table, vstack
from docopt import docopt
from glob import glob
import os
import re


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


def entry(inputs, output):
    if len(inputs) < 2:
        raise AttributeError('digicam-concatenate must take 1 output and at '
                             'least 2 inputs files as arguments')
    tables = [Table.read(input) for input in inputs]
    result = vstack(tables)
    if os.path.isfile(output):
        os.remove(output)
    result.write(output)


if __name__ == '__main__':
    args = docopt(__doc__)
    inputs = args['<INPUTS>']
    if len(inputs) == 1:
        inputs = glob(inputs[0])
        inputs.sort(key=alphanum_key)
    output = args['<OUTPUT>']
    entry(inputs, output)
