import numpy as np
from digicampipe.visualization import plot

directory = '/home/alispach/data/CRAB_01/'
hillas_filename = directory + 'hillas_all.npz'

hillas = np.load(hillas_filename)

cut_size = 10000
cut_width_length = 0.4


#mask = hillas['size'] > cut_size
mask = (hillas['width']/hillas['length']) < cut_width_length
mask *= (hillas['kurtosis'] > 0) * (hillas['kurtosis'] < 6)

for key, val in hillas.items():

    mask *= ~np.isnan(val)

print(np.argmax(hillas['kurtosis']))


plot.plot_hillas(hillas_dict=hillas, bins=20, mask=mask, title='Crab, $N_{showers} = %d$' %(np.sum(mask)))