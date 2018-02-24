'''
Usage:
  read_raw_events <files>...

Options:
  -h --help     Show this screen.
'''
from docopt import docopt
from digicampipe.io.event_stream import event_stream
from digicampipe.visualization import EventViewer


def entry():
    args = docopt(__doc__)

    display = EventViewer(
        event_stream(args['<files>']),
        scale='lin'
    )
    display.draw()
