'''
Look at events in the EventViewer
Usage:
  digicam-view [options] [--] <INPUT>...

Options:
  --baseline_16bits
'''
from docopt import docopt
from digicampipe.io import event_stream
from digicampipe.visualization import EventViewer


def entry():
    args = docopt(__doc__)
    data_stream = event_stream.event_stream(args['<INPUT>'],
                                            baseline_new=
                                            args['--baseline_16bits'])
    display = EventViewer(data_stream)
    display.draw()
