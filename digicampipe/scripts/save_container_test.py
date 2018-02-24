'''
Usage:
  dg_save_container_test [options] <files>...

Options:
  -h, --help  Show this help
'''
from docopt import docopt
from digicampipe.calib.camera import filter, random_triggers
from digicampipe.io import event_stream
from digicampipe.io.containers import save_to_pickle_gz


def entry():
    args = docopt(__doc__)

    unwanted_patch = [391, 392, 403, 404, 405, 416, 417]

    data_stream = event_stream(args['<files>'])
    data_stream = filter.fill_flag(data_stream, unwanted_patch=unwanted_patch)
    data_stream = random_triggers.fill_baseline_r0(data_stream, n_bins=3000)
    data_stream = filter.filter_missing_baseline(data_stream)
    save_to_pickle_gz(
        data_stream,
        'test.pickle',
        overwrite=True,
        max_events=100
    )
