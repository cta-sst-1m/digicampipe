import numpy as np
import matplotlib.pyplot as plt
from digicampipe.io.event_stream import event_stream
from digicampipe.visualization.plot import plot_array_camera
from eventio import IACTFile, EventIOFile, SimTelFile
from scipy.stats import exponnorm, moyal
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
import itertools
from digicampipe.visualization import EventViewer
from digicampipe.instrument.camera import DigiCam
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.hillas import hillas_parameters
from digicampipe.utils.likelihood import time_log_likelihood, \
    velocity_likelihood, gaussian_likelihood, combined_log_likelihood, \
    pulse_log_likelihood
from scipy.optimize import minimize
import astropy.units as u
from iminuit import Minuit

# files = ['/sst1m/MC/simtel_LHAASO/LHASSO-SST1M-20deg/gamma/gamma_E300._3.e3GeV_phi180_theta20_run20001.simtel.gz']
files = ['/sst1m/MC/simtel_krakow/gamma_100.0_300.0TeV_09.simtel.gz']
tel_id = 1
digicam = DigiCam
geom = digicam.geometry
pix_x = geom.pix_x
pix_y = geom.pix_y
template = NormalizedPulseTemplate.load('/home/alispach/Documents/PhD/'
                                        'ctasoft/digicampipe/digicampipe/tests/'
                                        'resources/pulse_SST-1M_pixel_0.dat')


def compute_true_parameters(event, geometry):
    true_pe = event['photoelectrons'][0]['photoelectrons']
    photoelectrons_times = event['photoelectrons'][0]['time']
    minimum_time = np.array([np.nanmin(t) for t in photoelectrons_times]).min()

    true_times = np.array([np.mean(t) for t in photoelectrons_times])
    true_hillas = hillas_parameters(geometry, true_pe)
    true_timing = timing_parameters(geometry, true_pe, true_times,
                                    true_hillas)
    return true_times, true_hillas, true_pe, true_timing


data_stream = event_stream(files)

viewer = EventViewer(data_stream)

with SimTelFile(files[0]) as f:
    for event in f:
        waveform = event['telescope_events'][tel_id]['adc_samples']
        n_channel, n_pixel, n_sample = waveform.shape
        baseline = event['camera_monitorings'][tel_id]['pedestal'] / n_sample
        waveform = waveform - baseline[..., np.newaxis]
        waveform = waveform[0]
        # print(baseline)
        waveform_integral = waveform.sum(axis=-1)
        photoelectrons = event['photoelectrons'][0]['photoelectrons']
        max_pixel = np.argmax(photoelectrons)
        times = np.arange(waveform.shape[-1]) * 4.
        plt.figure()
        plt.plot(times, waveform[max_pixel])
        photoelectrons_times = event['photoelectrons'][0]['time']
        mask = photoelectrons > 0


        time_minimizer = lambda x: -time_log_likelihood(times,
                                                        waveform,
                                                        x,
                                                        template=template)

        plt.figure()
        # t_fit = np.linspace(0, 200, num=200)
        # plt.plot(t_fit, time_minimizer(t_fit)[max_pixel])

        t0 = np.argmax(waveform, axis=-1)

        t0 = times[t0]
        bounds = [(10, 200) for _ in range(n_pixel)]
        # res = minimize(time_minimizer, x0=t0, bounds=bounds, method='Nelder-Mead')
        # t_reco = res.x

        mean_times = np.array([np.mean(t) for t in photoelectrons_times])
        mean_times = mean_times - np.nanmin(mean_times)

        true_hillas = hillas_parameters(geom, photoelectrons)
        true_timing = timing_parameters(geom, photoelectrons, mean_times,
                                        true_hillas)
        longi, trans = geom.get_shower_coordinates(
            true_hillas.x,
            true_hillas.y,
            true_hillas.psi
        )
        r = longi
        # r = np.sqrt(longi**2 + trans**2) * np.sign(longi)
        # sorted = np.argsort(r)
        # r = r[sorted]
        # mean_times = mean_times[sorted]
        # photoelectrons = photoelectrons[sorted]
        mask = photoelectrons > 0

        velociy_mini = lambda x: -velocity_likelihood(times, waveform[mask],
                                                      longi[mask], x[0], x[1],
                                                      template)
        res = minimize(velociy_mini, x0=[0.1, 50], bounds=[(0, 1), (0, 200)],
                       method='Nelder-Mead')
        print(res)
        plt.figure()
        plt.plot(longi, np.polyval(res.x, longi))
        plt.plot(longi, mean_times, linestyle='None', marker='o')

        space_mini = lambda x: -gaussian_likelihood(photoelectrons[mask],
                                                    pix_x[mask].value,
                                                    pix_y[mask].value,
                                                    x[0], x[1], x[2], x[3],
                                                    x[4])

        res = minimize(space_mini,
                       x0=[true_hillas.x.value, true_hillas.y.value,
                           true_hillas.width.value, true_hillas.length.value,
                           true_hillas.psi.value],
                       bounds=[(pix_x.value.min(), pix_x.value.max()),
                               (pix_y.value.min(), pix_y.value.max()),
                               ((true_hillas.width.value) * 0.5,
                                true_hillas.width.value * 2),
                               (true_hillas.length.value * 0.5,
                                true_hillas.length.value * 2),
                               (0, 2 * np.pi)],
                       method='Nelder-Mead')

        cam_display, _ = plot_array_camera(data=photoelectrons)
        cam_display.add_ellipse(centroid=(res.x[0], res.x[1]),
                                width=3 * res.x[2],
                                length=3 * res.x[3],
                                angle=res.x[4],
                                linewidth=7, color='r')

        p = np.polyfit(r[mask],
                       mean_times[mask],
                       deg=1,
                       w=np.sqrt(photoelectrons[mask]))

        predicted_time = np.polyval(p, r.value)
        plt.figure()
        plt.plot(longi.value, mean_times, label='data', marker='o', color='k',
                 linestyle='None')
        plt.plot(longi.value, predicted_time, label='fit', color='r')
        plt.xlabel('Longitudinal distance [mm]')
        plt.ylabel('Relative time [ns]')
        plt.legend()




        total_mini = lambda x: -combined_log_likelihood(times=times,
                                                      amplitudes=waveform[mask],
                                                      template=template,
                                                      pix_x=pix_x[mask].value,
                                                      pix_y=pix_y[mask].value,
                                                      charge=x[0],
                                                      baseline=0,
                                                      t_cm=x[1],
                                                      x_cm=x[2],
                                                      y_cm=x[3],
                                                      width=x[4],
                                                      length=x[5],
                                                      psi=x[6],
                                                      v=x[7],
                                                      sigma=10)

        total_mini_bis = lambda charge, t_cm, x_cm, y_cm, width, length, psi, v : -combined_log_likelihood(times=times,
                                                      amplitudes=waveform[mask],
                                                      template=template,
                                                      pix_x=pix_x[mask].value,
                                                      pix_y=pix_y[mask].value,
                                                      charge=charge,
                                                      baseline=0,
                                                      t_cm=t_cm,
                                                      x_cm=x_cm,
                                                      y_cm=y_cm,
                                                      width=width,
                                                      length=length,
                                                      psi=psi,
                                                      v=v,
                                                      sigma=10)

        m = Minuit(total_mini_bis, limit_charge=(0, 100000),
                                   limit_t_cm=(-50, 200),
                                   limit_x_cm=(2*pix_x.value.min(), 2*pix_x.value.max()),
                                   limit_y_cm=(2*pix_y.value.min(), 2*pix_y.value.max()),
                                   limit_width=(0, 500), limit_length=(0, 500),
                                   limit_psi=(0, 2*np.pi),
                                   limit_v=(-2, 2),
        charge=true_hillas.intensity*5, t_cm=true_timing.intercept,
                   x_cm=true_hillas.x.value, y_cm=true_hillas.y.value,
                   width=true_hillas.width.value, length=true_hillas.length.value,
                   psi=true_hillas.psi.value, v=true_timing.slope.value,
                   print_level=0)
        m.migrad()

        res = minimize(total_mini,
                       x0=[true_hillas.intensity*6, 20,
                           true_hillas.x.value+50, true_hillas.y.value,
                           true_hillas.width.value+63,
                           true_hillas.length.value, 1.2- true_hillas.psi.value, true_timing.slope.value],
                       bounds=[(0, 5000), (-50, 200),
                               (2*pix_x.value.min(), 2*pix_x.value.max()),
                               (2*pix_y.value.min(), 2*pix_y.value.max()),
                               (0.1, 500), (0.1, 500), (0, 2*np.pi), (0.1, 1)],
                       method='Powell',
                       options={'maxiter': 500})

        data = {'charge': res.x[0],
                't_cm': res.x[1],
                'x_cm': res.x[2],
                'y_cm': res.x[3],
                'width': res.x[4],
                'length': res.x[5],
                'psi': res.x[6],
                'v': res.x[7]}

        predicted_time = np.polyval([data['v'], data['t_cm']], longi.value)
        plt.figure()
        plt.plot(longi.value, mean_times, label='data', marker='o', color='k',
                 linestyle='None')
        plt.plot(longi.value, predicted_time, label='fit', color='r')
        plt.xlabel('Longitudinal distance [mm]')
        plt.ylabel('Relative time [ns]')
        plt.legend()

        plt.figure()
        plt.plot(trans.value, mean_times, label='data', marker='o', color='k',
                 linestyle='None')
        plt.xlabel('Transversal distance [mm]')
        plt.ylabel('Relative time [ns]')
        plt.legend()


        plt.figure()
        t_0 = np.argmin(mean_times)
        plt.plot(np.sqrt((trans.value - trans.value[t_0])**2 + (longi.value - longi.value[t_0])**2), mean_times, label='data', marker='o', color='k',
                 linestyle='None')
        plt.xlabel('distance [mm]')
        plt.ylabel('Relative time [ns]')
        plt.legend()

        cam_display, _ = plot_array_camera(data=waveform.max(axis=-1))
        cam_display.add_ellipse(centroid=(data['x_cm'], data['y_cm']),
                                width=3 * data['width'],
                                length=3 * data['length'],
                                angle=data['psi'],
                                linewidth=7, color='r')

        print(true_hillas, true_timing)

        print(data)
        print(m.values)

        viewer.draw()

        plt.show()
        break


time_shift = 10.66
charge = 10.235
baseline = 300.56
times = np.arange(0, 50, 1) * 4
amplitude = template(times - time_shift) * charge + baseline
# amplitude = template.amplitude * charge
# dt = np.diff(template.time)[0]
# samples_shift = int(time_shift / dt)
# amplitude = np.roll(amplitude, shift=samples_shift)
# times = template.time + time_shift
# mask = times > 40
# amplitude = amplitude[~mask]
# times = times[~mask]

time_minimizer = lambda x: -pulse_log_likelihood(times=times,
                                                 amplitudes=amplitude,
                                                 baseline=x[0],
                                                 charge=x[1],
                                                 t_0=x[2],
                                                 template=template)
x0 = [np.min(amplitude), np.max(amplitude) - np.min(amplitude),
      times[np.argmax(amplitude)]]
res = minimize(time_minimizer, x0=x0, method='trust-constr')
print(res)

reco_time = res.x[2]
reco_charge = res.x[1]
reco_baseline = res.x[0]
print('Likelihood : ', np.exp(-time_minimizer(res.x)))

time_fit = np.arange(-200, 200, 0.005)
llh = [pulse_log_likelihood(times=times, amplitudes=amplitude,
                            baseline=reco_baseline,
                            charge=reco_charge,
                            t_0=t,
                            template=template) for t in time_fit]
llh = np.array(llh)

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(time_fit, llh)
axes.axvline(reco_time, linestyle='--', color='k',
             label='Fitted time {}'.format(reco_time))
axes.axvline(time_shift, linestyle='--', color='b',
             label='Injected time {}'.format(time_shift))
axes.set_xlabel('t [ns]')
axes.set_ylabel('$\ln \mathcal{L}$')
axes.legend(loc='best')

cs = np.linspace(-amplitude.max(), amplitude.max(), num=1000)
llh = [pulse_log_likelihood(times=times, amplitudes=amplitude,
                            baseline=reco_baseline,
                            charge=c,
                            t_0=reco_time,
                            template=template) for c in cs]

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(cs, llh)
axes.axvline(reco_charge, linestyle='--', color='k',
             label='Fitted charge {}'.format(reco_charge))
axes.axvline(charge, linestyle='--', color='b',
             label='Injected charge {}'.format(charge))
axes.set_xlabel('Charge [LSB]')
axes.set_ylabel('$\ln \mathcal{L}$')
axes.legend(loc='best')

bs = np.linspace(-amplitude.max(), amplitude.max(), num=1000)
llh = [pulse_log_likelihood(times=times, amplitudes=amplitude, baseline=b,
                            charge=reco_charge,
                            t_0=reco_time,
                            template=template) for b in bs]

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(bs, llh)
axes.axvline(reco_baseline, linestyle='--', color='k',
             label='Fitted baseline {}'.format(reco_baseline))
axes.axvline(baseline, linestyle='--', color='b',
             label='Injected baseline {}'.format(baseline))
axes.set_xlabel('Baseline [LSB]')
axes.set_ylabel('$\ln \mathcal{L}$')
axes.legend(loc='best')

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(times, amplitude, label='data', marker='o', color='k',
          linestyle='None')
x = np.linspace(times.min(), times.max(), num=int(100 * len(times)))
axes.plot(x, template(x, t_0=reco_time) * reco_charge + reco_baseline,
          color='r',
          label='Fit : time = {:.2f} [ns], amplitude = {:.2f} [LSB], baseline = {:.2f}'.format(
              reco_time, reco_charge, reco_baseline))
axes.axvline(time_shift, linestyle='--', color='k',
             label='Injected time {} [ns]'.format(time_shift))
axes.axhline(charge + baseline, linestyle='--', color='b',
             label='Injected amplitude {} [LSB]'.format(charge))
axes.axhline(baseline, linestyle='--', color='r',
             label='Injected baseline {} [LSB]'.format(baseline))
axes.set_xlabel('t [ns]')
axes.set_ylabel('LSB')
axes.legend(loc='best')

plt.show()
exit()
