'''
Look at events in the EventViewer
Usage:
  digicamview <file>...
'''
from docopt import docopt
from digicampipe.io import event_stream
from digicampipe.visualization import EventViewer


def entry():
    args = docopt(__doc__)
    data_stream = event_stream.event_stream(args['<file>'], baseline_new=True)
    display = EventViewer(data_stream)
    display.draw()
