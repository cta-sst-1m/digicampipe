from datetime import timedelta
import numpy as np
from collections import OrderedDict
import pandas as pd

from functools import lru_cache
from glob import glob
from astropy import table
from os import path


class AuxService:
    def __init__(self, name, basepath):
        self.name = name
        self.basepath = basepath
        self.glob_expr = path.join(
            basepath,
            '{name}_{{date}}*.fits'.format(
                name=name,
            )
        )

    # maxsize needs to be > number of Services
    # lru cache for instance methods is shit...
    @lru_cache(maxsize=20)
    def at_date(self, date):
        ''' fetch fits Table for named aux service at date.
        If several files: append them in order.
        takes some time, result will maybe be cached.
        '''
        paths = sorted(
            glob(
                self.glob_expr.format(
                    date=date.strftime('%Y%m%d')
                )
            )
        )
        tables = []
        for p in paths:
            t = table.Table.read(p)
            if 'TIMESTAMP' in t.colnames:
                t.rename_column('TIMESTAMP', 'timestamp')
            tables.append(t)

        merged_table = table.vstack(tables, metadata_conflicts='silent')
        merged_table.meta = combine_table_metas(tables)
        merged_table = merged_table.filled()
        return merged_table

    def at(self, event_timestamp_in_ns):
        datetime = pd.to_datetime(event_timestamp_in_ns, unit='ns')
        date = (datetime - timedelta(hours=12)).date()
        table = self.at_date(date)
        table_index = np.searchsorted(
            table['timestamp'],
            event_timestamp_in_ns/1e6
        )
        # this row is still a astropy.Table.Row thing
        row = table[table_index - 1]
        # the return value is a numpy.recarray view into the row.
        # it allows for direct "."-attribute access
        return np.array(row).view(np.recarray)


def combine_table_metas(tables):
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
                    result[k] = set([v])
                else:
                    result[k].add(v)

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
                print('len{(k)} is not 1', k)
            result[k] = v.pop()
    return result
