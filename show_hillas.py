import numpy as np
from digicampipe.visualization import plot

directory = '/home/alispach/data/CRAB_01/'
hillas_filename = directory + 'hillas.npz'

hillas = np.load(hillas_filename)

plot.plot_hillas(hillas_dict=hillas, bins=30)