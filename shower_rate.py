'''
Usage:
  shower_rate <hillas_text_file>
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)

    hillas = np.genfromtxt(
        args['<hillas_text_file>'],
        names=[
            'size', 'cen_x', 'cen_y', 'length',
            'width', 'r', 'phi', 'psi',
            'miss', 'alpha', 'skewness',
            'kurtosis', 'event_number', 'time_stamp'
        ]
    )

    time = hillas['time_stamp']
    time = np.sort(time)
    print(np.diff(time))

    plt.figure()
    plt.title('Cherenkov rate')
    hist = plt.hist(
        np.diff(time),
        bins=100,
        log=True,
        normed=False,
        align='mid'
    )
    n_entries = np.sum(hist[0])
    bin_width = hist[1][1] - hist[1][0]
    param = expon.fit(np.diff(time), floc=0)
    pdf_fit = expon(loc=param[0], scale=param[1])
    plt.plot(
        hist[1],
        n_entries * bin_width * pdf_fit.pdf(hist[1]),
        label='$f_{trigger}$ = %0.2f [Hz]' % (1E9 / param[1]),
        linestyle='--'
    )
    plt.xlabel('$\Delta t$ [ns]')
    plt.legend(loc='best')
    plt.show()
