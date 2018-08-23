'''
Look at events in the EventViewer
Usage:
  digicam-view [options] [--] <INPUT>...

Options:
  --baseline_16bits
  --start=N         Event to skip
                    [Default: 0]
'''
from docopt import docopt
from digicampipe.io import event_stream
from digicampipe.visualization import EventViewer


def entry():
    args = docopt(__doc__)
    data_stream = event_stream.event_stream(args['<INPUT>'])
    for _, i in zip(data_stream, range(int(args['--start']))):

        pass
    display = EventViewer(data_stream)
    display.draw()
