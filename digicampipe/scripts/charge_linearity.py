from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.charge import \
    compute_charge_with_saturation_and_threshold
from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz' \
         ''.format(i) for i in range(1350, 1454 + 1, 1)]
# files = files[50:52]
ac_levels = np.hstack([np.arange(0, 20, 1), np.arange(20, 450, 5)])
n_pixels = 1296
n_files = len(files)
filename = 'charge_linearity_final.npz'

debug = False

shape = (n_files, n_pixels)
amplitude_mean = np.zeros(shape)
amplitude_std = np.zeros(shape)
charge_mean = np.zeros(shape)
charge_std = np.zeros(shape)

for i, file in tqdm(enumerate(files), total=n_files):

    events = calibration_event_stream(file)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    # events = compute_charge_with_saturation(events, integral_width=7)
    events = compute_charge_with_saturation_and_threshold(events,
                                                          integral_width=7,
                                                          debug=debug)
    # events = compute_maximal_charge(events)

    for n, event in enumerate(events):

        charge_mean[i] += event.data.reconstructed_charge
        amplitude_mean[i] += event.data.reconstructed_amplitude

        charge_std[i] += event.data.reconstructed_charge**2
        amplitude_std[i] += event.data.reconstructed_amplitude**2

    charge_mean[i] = charge_mean[i] / (n + 1)
    charge_std[i] = charge_std[i] / (n + 1)
    charge_std[i] = np.sqrt(charge_std[i] - charge_mean[i]**2)
    amplitude_mean[i] = amplitude_mean[i] / (n + 1)
    amplitude_std[i] = amplitude_std[i] / (n + 1)
    amplitude_std[i] = np.sqrt(amplitude_std[i] - amplitude_mean[i]**2)

np.savez(filename, charge_mean=charge_mean, charge_std=charge_std,
         amplitude_mean=amplitude_mean, amplitude_std=amplitude_std,
         ac_levels=ac_levels)

plt.figure()
plt.scatter(charge_mean.ravel(), amplitude_mean.ravel())


x = np.arange(1, charge_mean.max().astype(int))
plt.figure()
plt.scatter(charge_mean.ravel(), charge_std.ravel()/charge_mean.ravel())
plt.scatter(amplitude_mean.ravel(),
            amplitude_std.ravel()/amplitude_mean.ravel())
plt.yscale('log')
plt.xscale('log')
plt.plot(x, 1/np.sqrt(x))

plt.show()
