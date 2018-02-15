'''
Something with baseline stability

Usage:
  baseline_stability <fitsfile>

Options:
  -h --help     Show this screen.
'''
from digicampipe.io.event_stream import event_stream
import matplotlib.pyplot as plt
from docopt import docopt


def main(infile):

    baselines = []
    for data in event_stream(infile):
        for tel_id in data.r0.tels_with_data:
            baselines.append(data.r0.tel[tel_id].baseline)

    plt.figure()
    plt.plot(baselines[0])
    plt.plot(baselines[1])
    plt.plot(baselines[2])
    plt.plot(baselines[-1])
    plt.show()

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
