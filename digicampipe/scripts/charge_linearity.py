from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.charge import compute_charge_with_saturation, \
    compute_amplitude, compute_charge_with_saturation_and_threshold
from digicampipe.calib.peak import find_pulse_with_max
from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
import numpy as np
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D
from tqdm import tqdm


files = ['/sst1m/raw/2018/06/28/SST1M_01/' \
         'SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1420, 1454+1)]
files = files
ac_levels = np.hstack([np.arange(0, 20, 1), np.arange(20, 450, 5)])
n_pixels = 1296
n_files = len(files)

debug = False

for i, file in tqdm(enumerate(files), total=n_files):

    events = calibration_event_stream(file)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    # events = compute_charge_with_saturation(events, integral_width=7)
    events = compute_charge_with_saturation_and_threshold(events,
                                                          integral_width=7,
                                                          debug=debug)
    events = find_pulse_with_max(events)
    events = compute_amplitude(events)
    # events = compute_maximal_charge(events)

    charge_histo = Histogram1D(data_shape=(1296,),
                               bin_edges=np.arange(-200, 4095 * 50, 1))
    amplitude_histo = Histogram1D(data_shape=(1296,),
                                  bin_edges=np.arange(-200, 4095, 1/16))

    amplitude = np.zeros((n_files, n_pixels))
    amplitude_std = np.zeros((n_files, n_pixels))
    charge = np.zeros((n_files, n_pixels))
    charge_std = np.zeros((n_files, n_pixels))
    for event in events:

        c = event.data.reconstructed_charge.reshape(-1, 1)
        a = event.data.reconstructed_amplitude
        charge_histo.fill(c)
        amplitude_histo.fill(a)

    charge[i] = charge_histo.mean()
    charge_std[i] = charge_histo.std()
    amplitude[i] = amplitude_histo.mean()
    amplitude_std[i] = amplitude_histo.std()

np.savez('test.npz', charge=charge, charge_std=charge_std,
         amplitude=amplitude,
         amplitude_std=amplitude_std, ac_levels=ac_levels)


pixel = 0
plt.figure()
plt.errorbar(charge[:, pixel], amplitude[:, pixel], xerr=charge_std[:, pixel],
             yerr=amplitude_std[:, pixel])
plt.show()
