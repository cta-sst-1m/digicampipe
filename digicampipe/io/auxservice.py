from collections import OrderedDict, namedtuple
from datetime import timedelta
from functools import lru_cache
from glob import glob
from os import path
from warnings import warn

import numpy as np
import pandas as pd
from astropy import table


class AuxService:
    def __init__(self, name, basepath):
        self.name = name
        self.basepath = basepath
        self.glob_expr_fits = path.join(
            basepath,
            '{name}_{{date}}*.fits'.format(
                name=name,
            )
        )
        self.glob_expr_fits_gz = path.join(
            basepath,
            '{name}_{{date}}*.fits.gz'.format(
                name=name,
            )
        )
        self.namedtuple_klass = None

    def get_paths(self, date):
        fits_files = glob(
            self.glob_expr_fits.format(
                date=date.strftime('%Y%m%d')
            )
        )
        fits_gz_files = glob(
            self.glob_expr_fits_gz.format(
                date=date.strftime('%Y%m%d')
            )
        )
        fits_files.extend(fits_gz_files)
        return sorted(fits_files)

    # maxsize needs to be > number of Services
    # lru cache for instance methods is shit...
    @lru_cache(maxsize=20)
    def at_date(self, date):
        ''' fetch fits Table for named aux service at date.
        If several files: append them in order.
        takes some time, result will maybe be cached.
        '''
        paths = self.get_paths(date)
        if len(paths) == 0:
            raise RuntimeError("no data found for " + self.name + " on " +
                               str(date))
        combined_table = combine_tables(paths)

        # side effect!
        # we've just read a new day, so we update the format of our
        # return value
        self.namedtuple_klass = namedtuple(
            self.name + "Row",
            combined_table.colnames
        )
        return combined_table

    def at(self, event_timestamp_in_ns):
        datetime = pd.to_datetime(event_timestamp_in_ns, unit='ns')
        date = (datetime - timedelta(hours=12)).date()
        table = self.at_date(date)

        event_timestamp_in_ms = event_timestamp_in_ns / 1e6
        table_index = np.searchsorted(
            table['timestamp'],
            event_timestamp_in_ms
        )
        # this row is still a astropy.Table.Row thing
        row = table[table_index - 1]
        return self.namedtuple_klass(**{
            name: row[name]
            for name in table.colnames
        })


def read_table(path):
    ''' basically astropy.table.Table.read(path), but
    we need a "timestamp" column to syncronize with event times.
    In some files "timestamp" is called "TIMESTAMP" so we rename them.
    '''
    t = table.Table.read(path)
    if 'TIMESTAMP' in t.colnames:
        t.rename_column('TIMESTAMP', 'timestamp')
    return t


def combine_tables(paths):
    ''' merge astropy.table.Tables read from paths

    the meta information from the tables is merged in a complex way, c.f.
    combine_table_metas
    '''
    tables = [read_table(path) for path in paths]
    merged_table = table.vstack(tables, metadata_conflicts='silent')
    merged_table.meta = combine_table_metas(tables)
    return merged_table


def combine_table_metas(tables):
    ''' combine meta information (i.e. table headers) from fits tables.
    CHECKSUM/DATASUM:
        these must be wrong for the combined table, so they are dropped
    TSTART/TSTOP: the minimum of TSTART and the maximum of TSTOP are used.
    TELAPSE: the sum of all individual TELAPSE is used.
    FILENAME: the (lexical) minimum is used and the word "combined" is
            put into the name.
    all others:
        If they are all identical: the merged meta, will get this value.
        If not ... we get a random value of all those, we found. :-(
    '''
    metas = [t.meta for t in tables]
    result = OrderedDict()
    drop_keys = [
        'DATASUM',
        'CHECKSUM',
    ]
    list_keys = [
        'FILENAME',
        'TSTART',
        'TSTOP',
        'TELAPSE',
    ]
    for m in metas:
        for k, v in m.items():
            if k in drop_keys:
                continue
            if k in list_keys:
                if k not in result:
                    result[k] = [v]
                else:
                    result[k].append(v)
            else:
                if k not in result:
                    result[k] = [v]
                else:
                    result[k].append(v)

    for k, v in result.items():
        if k == 'FILENAME':
            name = min(v)
            name = name[:-8] + 'combined' + name[-5:]
            result[k] = name
            continue
        if k == 'TELAPSE':
            result[k] = np.sum(v)
            continue
        if k == 'TSTART':
            result[k] = np.min(v)
            continue
        if k == 'TSTOP':
            result[k] = np.max(v)
            continue
        else:
            if len(v) != 1:
                warn(k + ' has ' + str(len(v)) + ' data points instead of 1')
            result[k] = v.pop()
    return result
