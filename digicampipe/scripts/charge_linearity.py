from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.charge import \
    compute_charge_with_saturation_and_threshold
from digicampipe.utils.pulse_template import NormalizedPulseTemplate

from digicampipe.utils.led import ACLEDInterpolator
from digicampipe.calib.baseline import subtract_baseline, fill_digicam_baseline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d


def compute_linearity(measured_pe, true_pe):

    n_pixels = true_pe.shape[1]
    linearity_func = []

    for i in range(n_pixels):

        temp = interp1d(measured_pe[:, i], true_pe[:, i] / measured_pe[:, i])
        linearity_func.append(temp)

    return linearity_func


data = np.load('charge_linearity_final.npz')
charge = data['charge_mean'] - data['charge_mean'][0]
amplitude = data['amplitude_mean'] - data['amplitude_mean'][0]
std_charge = data['charge_std']
std_amplitude = data['amplitude_std']
ac_leds = np.load('/home/alispach/data/tests/mpe/mpe_fit_results.npz')
ac_levels = ac_leds['ac_levels']
gain = np.nanmean(ac_leds['gain'], axis=0)
pe = ac_leds['mu'] - ac_leds['mu'][0]
xt = np.nanmean(ac_leds['mu_xt'], axis=0)
pe_err = ac_leds['mu_error']

template = NormalizedPulseTemplate.load('/home/alispach/ctasoft/digicampipe/'
                                        'digicampipe/tests/resources/'
                                        'pulse_SST-1M_pixel_0.dat')
ratio = template.compute_charge_amplitude_ratio(7, 4)
print(ratio)
pe_charge = charge / gain * (1 - xt)
pe_amplitude = amplitude / (gain * ratio) * (1 - xt)

pe_for_calib = pe_amplitude.copy()
pe_for_calib[pe_for_calib < 100] = pe[pe_for_calib < 100]
pe_for_calib[pe_for_calib > 500] = np.nan
ac_led = ACLEDInterpolator(ac_levels, pe_for_calib)

true_pe = ac_led(ac_levels).T

pe_std_charge = std_charge / gain * (1 - xt)
pe_std_amplitude = std_amplitude / (gain * ratio) * (1 - xt)

linearity_charge = compute_linearity(charge, true_pe)
linearity_amplitude = compute_linearity(amplitude, true_pe)

plt.figure()
plt.plot(true_pe[:, 0], 1 / linearity_charge[0](pe_charge[:, 0]))
plt.plot(true_pe[:, 0], 1 / linearity_amplitude[0](pe_amplitude[:, 0]))
plt.xscale('log')
plt.yscale('log')
plt.xlim(10, 2000)
plt.xlabel('Number of p.e.')
plt.ylabel('Gain [LSB / p.e.]')
plt.show()

files = ['/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{}.fits.fz' \
         ''.format(i) for i in range(1350, 1454 + 1, 1)]
files = files[100:102]
ac_levels = np.hstack([np.arange(0, 20, 1), np.arange(20, 450, 5)])
n_pixels = 1296
n_files = len(files)
filename = 'charge_linearity_final_final.npz'
debug = False
shape = (n_files, n_pixels)
amplitude_mean = np.zeros(shape)
amplitude_std = np.zeros(shape)
charge_mean = np.zeros(shape)
charge_std = np.zeros(shape)
pe_mean = np.zeros(shape)
pe_std = np.zeros(shape)

for i, file in tqdm(enumerate(files), total=n_files):

    events = calibration_event_stream(file)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    # events = compute_charge_with_saturation(events, integral_width=7)
    events = compute_charge_with_saturation_and_threshold(events,
                                                          integral_width=7,
                                                          debug=debug,
                                                          linearity=linearity_charge,
                                                          measured_charge=charge,
                                                          true_charge=true_pe)
    # events = compute_maximal_charge(events)

    for n, event in enumerate(events):

        charge_mean[i] += event.data.reconstructed_charge
        amplitude_mean[i] += event.data.reconstructed_amplitude
        pe_mean[i] += event.data.reconstructed_number_of_pe

        charge_std[i] += event.data.reconstructed_charge**2
        amplitude_std[i] += event.data.reconstructed_amplitude**2
        pe_std[i] += event.data.reconstructed_number_of_pe**2

    charge_mean[i] = charge_mean[i] / (n + 1)
    charge_std[i] = charge_std[i] / (n + 1)
    charge_std[i] = np.sqrt(charge_std[i] - charge_mean[i]**2)
    amplitude_mean[i] = amplitude_mean[i] / (n + 1)
    amplitude_std[i] = amplitude_std[i] / (n + 1)
    amplitude_std[i] = np.sqrt(amplitude_std[i] - amplitude_mean[i]**2)

    pe_mean[i] = pe_mean[i] / (n + 1)
    pe_std[i] = pe_std[i] / (n + 1)
    pe_std[i] = np.sqrt(pe_std[i] - pe_mean[i]**2)

np.savez(filename, charge_mean=charge_mean, charge_std=charge_std,
         amplitude_mean=amplitude_mean, amplitude_std=amplitude_std,
         ac_levels=ac_levels, pe_mean=pe_mean, pe_std=pe_std)

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

plt.figure()
plt.scatter(pe_mean.ravel(), pe_std.ravel()/pe_mean.ravel(), s=1, color='k')
plt.yscale('log')
plt.xscale('log')
plt.plot(x, 1/np.sqrt(x))

plt.show()
