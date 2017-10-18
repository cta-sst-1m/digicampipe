import numpy as np
from digicampipe.visualization import plot
import plot_alpha_corrected
import matplotlib.pyplot as plt

directory = '/home/alispach/data/CRAB_01/'
hillas_filename = directory + 'hillas.npz'

hillas = dict(np.load(hillas_filename))

cut_size = 10000
cut_width_length = 0.5
cut_r = 350

mask = np.ones(hillas['size'].shape[0], dtype=bool)
mask *= (hillas['kurtosis'] > 0) * (hillas['kurtosis'] < 6)

source_xs = [0]
source_ys = [0]

alpha_max = 0

for source_x in source_xs:
    for source_y in source_ys:

        hillas_cor = plot_alpha_corrected.correct_alpha(hillas, source_x=source_x, source_y=source_y)

        alpha_histo = np.histogram(hillas_cor['alpha'], bins=20)

        alpha_0_count = alpha_histo[0][0]
        print(alpha_0_count)

        if alpha_0_count >= alpha_max:

            print(alpha_max)
            alpha_max = alpha_0_count
            true_source = [source_x, source_y]


print(true_source, alpha_max)

# hillas_corr = plot_alpha_corrected.correct_alpha(hillas, source_x=true_source[0], source_y=true_source[1])
plot.plot_hillas(hillas_dict=hillas, bins='auto')#, title='Crab (%0.1f, %0.1f)' %(true_source[0], true_source[1]))

plt.figure()
plt.hist(hillas['time_spread'], bins='auto')
plt.xlabel('time spread [ns]')
plt.show()
