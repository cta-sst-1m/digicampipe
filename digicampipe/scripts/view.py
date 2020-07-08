'''
Look at events in the EventViewer
Usage:
  digicam-view [options] [--] <INPUT>...

Options:
  --start=N         Event to skip
                    [Default: 0]
  --event_id=N      Event id to start
                    [Default: None]
'''
from docopt import docopt

from digicampipe.io import event_stream
from digicampipe.visualization import EventViewer


def entry():
    args = docopt(__doc__)

    event_id = args['--event_id']
    if event_id == 'None':

        event_id = None
    else:

        event_id = int(event_id)

    data_stream = event_stream.event_stream(args['<INPUT>'],
                                            event_id=event_id)
    for _, i in zip(data_stream, range(int(args['--start']))):
        pass
    display = EventViewer(data_stream)
    display.draw()


if __name__ == '__main__':
    entry()