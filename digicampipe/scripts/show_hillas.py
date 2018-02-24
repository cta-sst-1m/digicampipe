'''
Usage:
  dg_show_hillas [options] <file>

Options:
  -h, --help    Show this help
'''
import numpy as np
import plot_alpha_corrected
import matplotlib.pyplot as plt
from digicampipe.visualization import plot
from docopt import docopt


def entry():
    args = docopt(__doc__)
    hillas = dict(np.load(args['<file>']))

    source_xs = [0]
    source_ys = [0]

    alpha_max = 0

    for source_x in source_xs:
        for source_y in source_ys:

            hillas_cor = plot_alpha_corrected.correct_alpha(
                hillas,
                source_x=source_x,
                source_y=source_y
            )

            alpha_histo = np.histogram(hillas_cor['alpha'], bins=30)

            alpha_0_count = alpha_histo[0][0]
            print(alpha_0_count)

            if alpha_0_count >= alpha_max:
                print(alpha_max)
                alpha_max = alpha_0_count
                true_source = [source_x, source_y]

    print(true_source, alpha_max)
    plot.plot_hillas(hillas_dict=hillas, bins='auto')

    hillas['time_spread'] = hillas['time_spread'][np.isfinite(hillas['time_spread'])]

    plt.figure()
    plt.hist(hillas['time_spread'], bins='auto')
    plt.xlabel('time spread [ns]')
    plt.show()
