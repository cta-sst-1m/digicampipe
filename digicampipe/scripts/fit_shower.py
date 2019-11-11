from digicampipe.utils.fitter import ShowerFitter
from ctapipe.image import hillas_parameters
from ctapipe.image.timing_parameters import timing_parameters
from pkg_resources import resource_filename
import os
import numpy as np
import matplotlib.pyplot as plt
from digicampipe.io.event_stream import event_stream
from digicampipe.visualization import EventViewer
from digicampipe.visualization.plot import plot_array_camera
from eventio import SimTelFile
from digicampipe.instrument.camera import DigiCam
from copy import copy

files = ['/sst1m/MC/simtel_krakow/gamma_100.0_300.0TeV_09.simtel.gz']
tel_id = 1
geom = DigiCam.geometry


def compute_true_parameters(event, geometry):
    true_pe = event['photoelectrons'][0]['photoelectrons']
    photoelectrons_times = event['photoelectrons'][0]['time']
    minimum_time = np.array([np.nanmin(t) for t in photoelectrons_times]).min()

    true_times = np.array([np.mean(t) for t in photoelectrons_times])
    true_hillas = hillas_parameters(geometry, true_pe)
    true_timing = timing_parameters(geometry, true_pe, true_times,
                                    true_hillas)
    return true_times, true_hillas, true_pe, true_timing


# data_stream = event_stream(files)

# viewer = EventViewer(data_stream)
# viewer.draw()
debug = True

mc_values = {'charge': [],
             't_cm': [],
              'x_cm': [],
              'y_cm': [],
              'width': [],
              'length': [],
              'psi': [],
              'v': [],
        }

standard_values = {'charge': [],
             't_cm': [],
              'x_cm': [],
              'y_cm': [],
              'width': [],
              'length': [],
              'psi': [],
              'v': [],
        }
fit_values = {'charge': [],
             't_cm': [],
              'x_cm': [],
              'y_cm': [],
              'width': [],
              'length': [],
              'psi': [],
              'v': [],
        }
n_events = 800


with SimTelFile(files[0]) as f:
    for event, i in zip(f, range(n_events)):

        print(i)

        waveform = event['telescope_events'][tel_id]['adc_samples']
        n_channel, n_pixel, n_sample = waveform.shape
        baseline = event['camera_monitorings'][tel_id]['pedestal'] / n_sample
        waveform = waveform - baseline[..., np.newaxis]
        waveform = waveform[0]
        # print(baseline)
        waveform_integral = waveform.sum(axis=-1)
        photoelectrons = event['photoelectrons'][0]['photoelectrons']
        photoelectrons_times = event['photoelectrons'][0]['time']
        mean_times = np.array([np.mean(t) for t in photoelectrons_times])
        mean_times = mean_times - np.nanmin(mean_times)

        max_pixel = np.argmax(photoelectrons)
        times = np.arange(waveform.shape[-1]) * 4.

        fitter = ShowerFitter(data=waveform)
        method = 'Powell'
        # method = 'TNC'
        fitter.fit(method=method, verbose=False)

        true_hillas = hillas_parameters(geom, photoelectrons)
        true_timing = timing_parameters(geom, photoelectrons, mean_times,
                                        true_hillas)

        mc_values['charge'].append(true_hillas.intensity)
        mc_values['x_cm'].append(true_hillas.x.value)
        mc_values['y_cm'].append(true_hillas.y.value)
        mc_values['width'].append(true_hillas.width.value)
        mc_values['length'].append(true_hillas.length.value)
        mc_values['psi'].append(true_hillas.psi.value)
        mc_values['t_cm'].append(true_timing.intercept)
        mc_values['v'].append(true_timing.slope.value)



        for key, val in fitter.end_parameters.items():

            fit_values[key].append(val)
            standard_values[key].append(fitter.start_parameters[key])

        print(fitter)
        print(fit_values, standard_values, mc_values)

        if not debug:
            continue

        cam_display, _ = plot_array_camera(data=photoelectrons)
        cam_display.add_ellipse(centroid=(true_hillas.x.value, true_hillas.y.value),
                                width=3 * true_hillas.width.value,
                                length=3 * true_hillas.length.value,
                                angle=true_hillas.psi.value,
                                linewidth=7, color='r')

        cam_display, _ = plot_array_camera(data=waveform.max(axis=-1))
        cam_display.add_ellipse(centroid=(fitter.start_parameters['x_cm'],
                                          fitter.start_parameters['y_cm']),
                                width=3 * fitter.start_parameters['width'],
                                length=3 * fitter.start_parameters['length'],
                                angle=fitter.start_parameters['psi'],
                                linewidth=7, color='r')

        fitter.plot()

        mask = photoelectrons > 0
        dx = (geom.pix_x.value - fitter.end_parameters['x_cm'])
        dy = (geom.pix_y.value - fitter.end_parameters['y_cm'])
        long = dx * np.cos(fitter.end_parameters['psi']) + dy * np.sin(
            fitter.end_parameters['psi'])

        long = long[mask]

        x = np.linspace(long.min(), long.max(), num=100)
        y = np.polyval([fitter.end_parameters['v'], fitter.end_parameters['t_cm']],
                       x)
        plt.figure()
        plt.scatter(long, mean_times[mask], color='k')
        plt.plot(x, y, label='Fit', color='r')

        # for param in fitter.end_parameters.keys():
        #     fitter.plot_1dlikelihood(param)
        fitter.plot_likelihood('x_cm', 'y_cm', size=(100, 100))
        fitter.plot_likelihood('width', 'length', size=(100, 100))
        print(fitter)
        plt.show()


def dict_to_dictnumpy(dictionary):

    for key, val in dictionary.items():

        dictionary[key] = np.array(val)

    return dictionary

np.save('mc.npy', dict_to_dictnumpy(mc_values))
np.save('standard.npy', dict_to_dictnumpy(standard_values))
np.save('fit.npy', dict_to_dictnumpy(fit_values))
