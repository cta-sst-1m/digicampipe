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
import numpy as np

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
    if len(inputs) < 1:
        raise AttributeError('digicam-concatenate must take 1 output and at '
                             'least 1 input file as arguments')
    tables = []
    for input in inputs:
        if os.path.isfile(input):
            tables.append(Table.read(input))
        else:
            print('WARNING:', input, 'does not exist, skipping it.')
    columns_type = {}
    for table in tables:
        for column_idx in range(len(table.columns)):
            column = table.columns[column_idx]
            type = column.dtype
            if np.all(np.logical_or(column == 0, column==1)):
                type = bool
            if column.name in columns_type.keys():
                columns_type[column.name] = np.result_type(
                    type,
                    columns_type[column.name]
                )
            else:
                columns_type[column.name] = type
    tables_stackable = []
    for table in tables:
        columns_converted = []
        for key, val in columns_type.items():
            columns_converted.append(table.columns[key].astype(val))
        tables_stackable.append(Table(columns_converted))
    result = vstack(tables_stackable)
    if os.path.isfile(output):
        print('WARNING:', output, 'existed, overwriting it.')
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
