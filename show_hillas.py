import numpy as np
from digicampipe.visualization import plot
import plot_alpha_corrected
import matplotlib.pyplot as plt

directory = '/home/alispach/data/CRAB_01/'
hillas_filename = directory + 'hillas_crab.npz'
hillas_filename = directory + 'hillas_crab_10k_cut_100pe.npz'
hillas_filename = directory + 'hillas_crab_all_cut_100pe.npz'
hillas_filename = directory + 'hillas_crab_100_nogeom_changed_cut_100pe.npz'
hillas_filename = directory + 'hillas_crab_100_geom_changed_cut_100pe.npz'
hillas_filename = directory + 'Merge_hillas_wo_corr.txt'

# hillas = np.load(hillas_filename)

hillas = np.genfromtxt(hillas_filename, names=['size', 'cen_x', 'cen_y', 'length', 'width', 'r', 'phi', 'psi', 'miss', 'alpha', 'skewness', 'kurtosis', 'event_number', 'time_stamp'])

print(hillas['size'].shape)
# print(hillas.shape)
# print(hillas.reshape(14, hillas.shape[0]//14))

cut_size = 10000
cut_width_length = 0.5
cut_r = 350

mask = np.ones(hillas['size'].shape[0], dtype=bool)
#mask *= hillas['size'] > cut_size
#mask *= (hillas['width']/hillas['length']) < cut_width_length
mask *= (hillas['kurtosis'] > 0) * (hillas['kurtosis'] < 6)
#mask *= (hillas['r'] < cut_r)

#for key, val in hillas.items():

#    mask *= ~np.isnan(val)

(61.2, 7.5)

# source_xs = np.linspace(-300, 300, num=100)
# source_ys = np.linspace(-300, 300, num=100)
source_xs = [0] # np.linspace(-300, 300, num=50)
# source_xs = [61.2]#np.linspace(-300, 300, num=50)
source_ys = [0] # np.linspace(-300, 300, num=50)
# source_ys = [7.5]#np.linspace(-300, 300, num=50)

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

plt.show()



plt.show()
