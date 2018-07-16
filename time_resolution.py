


import matplotlib.pyplot as plt
from digicampipe.utils.utils import get_pulse_shape
import numpy as np
import h5py
from matplotlib import colors
from glob import glob
from scipy.interpolate import BSpline, CubicSpline


# We read the histograms into memory, so we have quick and easy access, also we define a few global variables like x_bin_center, y_bin_center and extent, for plotting and analysis of the histograms. We store the histograms in the H dict using the file names as keys.

# In[3]:


paths = sorted(glob('template_scan_dac_*.h5'))
H = {}
for path in paths:
    with h5py.File(path) as f:
        dset = f['adc_count_histo']
        H[path] = dset[...]
        extent = dset.attrs['extent']
        x_bin_edges = np.linspace(*extent[:2], dset.shape[1]+1)
        y_bin_edges = np.linspace(*extent[2:], dset.shape[2]+1)
        x_bin_center = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
        y_bin_center = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2


# The next function `analyse_2d_histo_for_pixel` takes one of these histograms we have just seen, calculates the profile (think TProfile, if you are a ROOT user) and fits a cubic spline to the profile (where we think we know it well enough).

# In[6]:


def analyse_2d_histo_for_pixel(histogram_2d):
    _h = histogram_2d
    N = _h.sum(axis=-1)
    mode = y_bin_center[_h.argmax(axis=-1)]
    mean = (_h * y_bin_center[None, :]).sum(axis=-1)  / N
    squared_sum = (y_bin_center[None, :] - mean[:, None])**2
    std = np.sqrt((_h * squared_sum).sum(axis=-1) / (N-1))

    average_std = np.nanmean(std)

    # For the spline we only use those bins, where we have "enough"
    # statistics. I define here "enough" as 100 entries
    G = N >= 100
    _x = x_bin_center[G]
    _y = mean[G]
    spl = CubicSpline(_x, _y)
    return {
        'mode': mode,
        'mean': mean,
        'std': std,
        'N': N,
        'spline': spl,
        'enough_entries': G,
    }


# The cell below tries to find the "best" spline for every pixel. You can see above that depending on the DAC setting, the pixel can saturate, which is visible here as a longer but flatter curve.
# 
# Other pixel look into LEDs which are comparatively dim, i.e. at low DAC settings these pixel might see no light at all, while at the highest DAC setting they see enough light to produce a nicely defined template curve.
# 
# In order to find the "best" (non-saturating) template I say:
#  * if all profiles have very low std deviations, then take the highest template.
#  * if not all profiles have low std deviations, then take the one with the smallest errors.
#  
# I think this method is not perfect, but at the moment, I have no better idea.

# In[9]:


splines = []
for pid in range(1296):
    sub_splines = {}
    for path, h in H.items():
        result = analyse_2d_histo_for_pixel(h[pid])
        max_amplitude = result['spline'](np.linspace(0, 20, 50)).max()
        sub_splines[(max_amplitude, np.nanmean(result['std']))] = result['spline']
    keys = list(sub_splines.keys())
    average_stds = np.array([k[-1] for k in keys])
    max_amplitudes = np.array([k[0] for k in keys])
    if (average_stds < 0.05).all():
        splines.append(sub_splines[keys[np.argmax(max_amplitudes)]])
    else:
        splines.append(sub_splines[keys[np.argmin(average_stds)]])


# The cell below simply plots the splines for all 1296 pixels into one plot, to understand if we really need one template per pixel

# In[10]:


x = []
y = []
_x = np.linspace(x_bin_center.min(), x_bin_center.max(), 1000)
for spl in splines:
    x.append(_x)
    y.append(spl(_x))
x = np.concatenate(x)
y = np.concatenate(y)
plt.figure(figsize=(18, 12))
histogram_2d, xe, ye, _ = plt.hist2d(
    x, 
    y, 
    bins=(501, 501), 
    range=[extent[:2], extent[2:]],
    norm=colors.LogNorm()
)
plt.grid()
plt.colorbar()




# Plotting the average pulse shape alone.

# In[13]:

histogram_2d[histogram_2d<np.max(histogram_2d)*.02] = 0
xc = (xe[1:] + xe[:-1]) / 2
yc = (ye[1:] + ye[:-1]) / 2
N = histogram_2d.sum(axis=-1)
mean = (histogram_2d * yc[None, :]).sum(axis=-1) / N
dy_mean_squared = (yc[None, :] - mean[:, None])**2
std = np.sqrt((histogram_2d * dy_mean_squared).sum(axis=-1) / N)
plt.figure(figsize=(14, 8))
plt.pcolormesh(xc, yc, np.log(histogram_2d.transpose()+1))
plt.errorbar(x=xc, y=mean, yerr=std, label='std of 1296 templates')
plt.plot(xc, mean, label='mean of 1296 templates')
plt.xlabel('time around 50% max height [ns]')
plt.ylabel('normalized amplitude')
plt.legend(loc='upper right')
plt.show()
