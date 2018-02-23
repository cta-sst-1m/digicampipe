'''
Usage:
  read_raw_events <files>...

Options:
  -h --help     Show this screen.
'''
from docopt import docopt
from digicampipe.io.event_stream import event_stream
from digicamviewer.viewer import EventViewer


def main(paths):
    data_stream = event_stream(paths)

    display = EventViewer(
        data_stream,
        n_samples=50,
        camera_config_file=digicam_config_file,
        scale='lin'
    )
    display.draw()


    for data in data_stream:
        pass


def entry():
    args = docopt(__doc__)
    main(paths=args['<files>'])
