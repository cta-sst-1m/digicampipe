import numpy as np
from digicampipe.instrument.light_source import ACLED
from tqdm import tqdm
import os

from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline, fill_dark_baseline, compute_baseline_shift
from digicampipe.calib.charge import \
    compute_charge_with_saturation_and_threshold, compute_number_of_pe_from_table, rescale_pulse
from digicampipe.io.event_stream import calibration_event_stream


def charge_to_pe(x, measured_average_charge, true_pe):

    X = measured_average_charge.T
    Y = true_pe.T

    dX = np.diff(X, axis=-1)
    dY = np.diff(Y, axis=-1)

    sign = np.sign(x)

    w = np.clip((np.abs(x[:, None]) - X[:, :-1]) / dX[:, :], 0, 1)

    y = Y[:, 0] + np.nansum(w * dY[:, :], axis=1)
    y = y * sign
    return y


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

# ac_levels = data_calib['ac_levels'][:, 0]
ac_levels = np.hstack([np.arange(0, 20, 2), np.arange(20, 450, 10)])
pde = 0.9 # window filter

pe = data_calib['mu']
pe_err = data_calib['mu_error']
ac_led = ACLED(ac_levels, pe, pe_err)
true_pe = ac_led(ac_levels).T * pde
# mask = true_pe < 5
# true_pe[mask] = pe[mask]

files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'. format() for i in range(1982, 2034 + 1, 1)]
# files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz'.format(i) for i in range(1350, 1454 + 1, 1)]
# files = files[100:]
# ac_levels = ac_levels[100:]
n_pixels = 1296
n_files = len(files)

assert n_files == len(ac_levels)
filename_1 = 'charge_linearity_24102018_dark.npz'
filename_2 = 'charge_resolution_24102018_125MHz.npz'

debug = False
pulse_tail = False
shape = (n_files, n_pixels)
nsb_mean = np.zeros(shape)
nsb_std = np.zeros(shape)
pe_mean = np.zeros(shape)
pe_std = np.zeros(shape)

timing = np.load('/sst1m/analyzed/calib/timing/timing.npz')
timing = timing['time'] // 4

charge_mean = np.load(filename_1)['charge_mean']
dark_baseline = charge_mean[0]
pe_interpolator = lambda x: charge_to_pe(x, charge_mean, true_pe)

for i, file in tqdm(enumerate(files), total=n_files):

    events = calibration_event_stream(file, max_events=max_events)
    events = fill_dark_baseline(events, dark_baseline)
    events = compute_baseline_shift(events)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    events = rescale_pulse(events, )
    # events = compute_charge_with_saturation(events, integral_width=7)
    events = compute_charge_with_saturation_and_threshold(events,
                                                          integral_width=integral_width,
                                                          debug=debug,
                                                          trigger_bin=timing,
                                                          saturation_threshold=saturation_threshold,
                                                          pulse_tail=pulse_tail)

    events = compute_number_of_pe_from_table(events, pe_interpolator)
    # events = compute_maximal_charge(events)

    for n, event in enumerate(events):

        pe_mean[i] += event.data.reconstructed_number_of_pe
        pe_std[i] += event.data.reconstructed_number_of_pe**2
        nsb_mean[i] += event.data.nsb_rate
        nsb_std[i] += event.data.nsb_rate**2

    pe_mean[i] = pe_mean[i] / (n + 1)
    nsb_mean[i] = nsb_mean[i] / (n + 1)
    pe_std[i] = pe_std[i] / (n + 1)
    pe_std[i] = np.sqrt(pe_std[i] - pe_mean[i]**2)
    nsb_std[i] = nsb_std[i] / (n + 1)
    nsb_std[i] = np.sqrt(nsb_std[i] - nsb_mean[i]**2)

np.savez(filename_2, pe_reco_mean=pe_mean, pe_reco_std=pe_std,
         ac_levels=ac_levels, pe=pe, pe_err=pe_err, true_pe=true_pe,
         nsb_mean=nsb_mean, nsb_std=nsb_std)

