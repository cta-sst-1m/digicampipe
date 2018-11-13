import numpy as np
from digicampipe.instrument.light_source import ACLED
from tqdm import tqdm
import os

from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
from digicampipe.calib.charge import \
    compute_charge_with_saturation_and_threshold, compute_number_of_pe_from_table
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.visualization.plot import plot_histo, plot_array_camera


integral_width = 7
# saturation_threshold = dict(np.load('/home/alispach/Documents/PhD/ctasoft/digicampipe/thresholds.npz'))
# saturation_threshold = saturation_threshold['threshold_charge']

# mean = np.nanmean(saturation_threshold)
# saturation_threshold[np.isnan(saturation_threshold)] = mean

saturation_threshold = 3000

max_events = None
directory = '/sst1m/analyzed/calib/mpe/'
file_calib = os.path.join(directory, 'mpe_fit_results_combined.npz')
data_calib = np.load(file_calib)

ac_levels = data_calib['ac_levels'][:, 0]
pe = data_calib['mu']
pe_err = data_calib['mu_error']
ac_led = ACLED(ac_levels, pe, pe_err)
true_pe = ac_led(ac_levels).T
# mask = true_pe < 5
# true_pe[mask] = pe[mask]


files = ['/home/alispach/Downloads/ac_scan/SST1M_01_20180628_{}.fits.fz' \
         ''.format(i) for i in range(1350, 1454 + 1, 1)]
# files = files[100:]
# ac_levels = ac_levels[100:]
n_pixels = 1296
n_files = len(files)

assert n_files == len(ac_levels)
filename_1 = 'charge_linearity_24102018_dark.npz'

debug = False
pulse_tail = False
shape = (n_files, n_pixels)
amplitude_mean = np.zeros(shape)
amplitude_std = np.zeros(shape)
baseline_mean = np.zeros(shape)
baseline_std = np.zeros(shape)
charge_mean = np.zeros(shape)
charge_std = np.zeros(shape)
pe_mean = np.zeros(shape)
pe_std = np.zeros(shape)

timing = np.load('/sst1m/analyzed/calib/timing/timing.npz')
timing = timing['time'] // 4

for i, file in tqdm(enumerate(files), total=n_files):

    events = calibration_event_stream(file, max_events=max_events)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    # events = compute_charge_with_saturation(events, integral_width=7)
    events = compute_charge_with_saturation_and_threshold(events,
                                                          integral_width=integral_width,
                                                          debug=debug,
                                                          trigger_bin=timing,
                                                          saturation_threshold=saturation_threshold,
                                                          pulse_tail=pulse_tail)
    # events = compute_maximal_charge(events)

    for n, event in enumerate(events):

        charge_mean[i] += event.data.reconstructed_charge
        amplitude_mean[i] += event.data.reconstructed_amplitude

        charge_std[i] += event.data.reconstructed_charge**2
        amplitude_std[i] += event.data.reconstructed_amplitude**2

        baseline_mean[i] += event.data.baseline
        baseline_std[i] += event.data.baseline**2

    charge_mean[i] = charge_mean[i] / (n + 1)
    charge_std[i] = charge_std[i] / (n + 1)
    charge_std[i] = np.sqrt(charge_std[i] - charge_mean[i]**2)
    amplitude_mean[i] = amplitude_mean[i] / (n + 1)
    amplitude_std[i] = amplitude_std[i] / (n + 1)
    amplitude_std[i] = np.sqrt(amplitude_std[i] - amplitude_mean[i]**2)
    baseline_mean[i] = baseline_mean[i] / (n + 1)
    baseline_std[i] = baseline_std[i] / (n + 1)
    baseline_std[i] = np.sqrt(baseline_std[i] - baseline_mean[i]**2)

np.savez(filename_1, charge_mean=charge_mean, charge_std=charge_std,
         amplitude_mean=amplitude_mean, amplitude_std=amplitude_std,
         ac_levels=ac_levels, pe=pe, pe_err=pe_err, baseline_mean=baseline_mean,
         baseline_std=baseline_std, true_pe=true_pe)
