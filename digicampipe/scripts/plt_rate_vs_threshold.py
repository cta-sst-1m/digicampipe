"""
Usage:
  plt_rate_vs_threshold <trigger_npz_file>...
"""
from os.path import split

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

args = docopt(__doc__)
fig = plt.figure()
for path in args['<trigger_npz_file>']:
    file_name = split(path)[-1][:-4]
    file = np.load(path)
    plt.errorbar(
        file['thresholds'],
        file['rate'] * 1E9,
        yerr=file['rate_error'] * 1E9,
        label=file_name)

plt.ylabel('rate [Hz]')
plt.xlabel('threshold [LSB]')
plt.yscale('log')
plt.grid(True)
plt.legend(loc='best')
plt.show()
