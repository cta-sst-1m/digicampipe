from digicamtoy.tracegenerator import NTraceGenerator, ShowerGenerator
from digicampipe.instrument.camera import DigiCam
import numpy as np
import warnings
from digicampipe.io.containers import DataContainer
from digicampipe.io.containers import CameraEventType
from digicampipe.visualization import EventViewer
from eventio import SimTelFile
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.timing_parameters import timing_parameters
import pandas as pd
import matplotlib.pyplot as plt


def compute_true_parameters(event, geometry):

    times = event['photoelectrons'][0]['time']
    # photoelectrons_times = pd.DataFrame(photoelectrons_times)
    # photoelectrons_times = photoelectrons_times - np.nanmin(photoelectrons_times) + 10
    # photoelectrons_times = photoelectrons_times.fillna(0)

    true_pe = []
    min_time = np.inf
    for t in times:

        if len(t):

            n = [1]*len(t)
        else:
            n = [0]
        min_time = min(min_time, np.min(t))
        true_pe.append(n)

    pe = event['photoelectrons'][0]['photoelectrons']
    true_times = np.array([np.mean(t) for t in times])
    true_hillas = hillas_parameters(geometry, pe)
    true_timing = timing_parameters(geometry, pe, true_times,
                                    true_hillas)

    return times, true_hillas, true_pe, true_timing

def digicamtoy_event_stream(
        generator,
        camera=DigiCam,
        event_id=None,
):
    """A generator that streams data from an HDF5 data file from DigicamToy
    Parameters
    ----------
    generator: NTraceGenerator
    max_events : int, optional
        maximum number of events to read
    camera : utils.Camera() default: utils.DigiCam
    chunk_size : Number of events to load into the memory at once
    event_id : TODO
    disable_bar: If set to true, the progress bar is not shown.
    """

    if event_id is not None:

        warnings.warn('The event id search is not implemented for this'
                      ' type of file')

    data = DataContainer()

    for event_id, event in enumerate(generator):

        data.r0.event_id = event_id
        data.r0.tels_with_data = [1, ]
        waveform = event.adc_count
        baseline = event.true_baseline
        n_channels, n_pixels, n_samples = 1, waveform.shape[0], waveform.shape[1]

        for tel_id in data.r0.tels_with_data:

            if event_id == 0:

                data.inst.num_channels[tel_id] = 1
                data.inst.num_pixels[tel_id] = n_pixels
                data.inst.geom[tel_id] = camera.geometry
                data.inst.cluster_matrix_7[tel_id] = camera.cluster_7_matrix
                data.inst.cluster_matrix_19[tel_id] = camera.cluster_19_matrix
                data.inst.patch_matrix[tel_id] = camera.patch_matrix
                data.inst.num_samples[tel_id] = n_samples
                data.r0.tel[tel_id].digicam_baseline = baseline
                data.r0.tel[tel_id].camera_event_type = \
                    CameraEventType.INTERNAL
                data.r0.tel[tel_id].array_event_type = CameraEventType.UNKNOWN

            data.r0.tel[tel_id].camera_event_number = event_id
            data.r0.tel[tel_id].local_camera_clock = event_id
            data.r0.tel[tel_id].gps_time = event_id
            data.r0.tel[tel_id].adc_samples = waveform

        yield data

template_filename = '/home/alispach/Documents/PhD/ctasoft/' \
                    'digicampipe/digicampipe/tests/resources/pulse_SST-1M_pixel_0.dat'

n_pixels = 1296
crosstalk = 0.08 * np.ones(n_pixels)
photon = np.zeros(n_pixels)
sigma_e = np.ones(n_pixels)
sigma_1 = np.ones(n_pixels)
gain = 5.6 * np.ones(n_pixels)
baseline = 200 * np.ones(n_pixels)
time_signal = 20 * np.ones(n_pixels)
jitter = 0.01
nsb_rate = 0.6 * np.ones(n_pixels)

params = {'time_start': 0, 'time_end': 200, 'time_sampling': 4,
          'n_pixels': 1296, 'crosstalk': crosstalk, 'gain_nsb': True,
          'n_photon': photon, 'poisson': False, 'sigma_e': sigma_e,
          'sigma_1': sigma_1, 'gain': gain, 'baseline': baseline,
          'time_signal': time_signal, 'jitter': jitter,
          'pulse_shape_file': template_filename, 'voltage_drop': False,
          'nsb_rate': nsb_rate, 'n_events': 10}

generator = ShowerGenerator(**params)
files = ['/sst1m/MC/simtel_krakow/gamma_100.0_300.0TeV_10.simtel.gz']
geom = DigiCam.geometry


with SimTelFile(files[0]) as f:

    for i, event_simtel in enumerate(f):

        if i < 10:

            continue

        pixel = 506
        simtel_times = event_simtel['photoelectrons'][0]['time'][pixel]
        simtel_times -= np.min(simtel_times)
        simtel_pe = event_simtel['photoelectrons'][0]['photoelectrons'][pixel]
        simtel_waveform = event_simtel['telescope_events'][1]['adc_samples'][0, pixel]
        n_samples = simtel_waveform.shape[-1]
        t_w = np.arange(n_samples) * 4

        true_times, true_hillas, true_pe, true_timing = compute_true_parameters(event_simtel, geom)
        generator.set_photoelectrons(true_times, true_pe, jitter=0.)
        data_stream = digicamtoy_event_stream(generator)
        for data in data_stream:

            toy_times = generator.time_signal[pixel]
            toy_pe = generator.n_photon[pixel]
            toy_waveform = data.r0.tel[1].adc_samples[pixel]
            pass

        plt.figure()
        plt.plot(t_w, simtel_waveform)
        for t in simtel_times:
            plt.axvline(t)
        print(len(simtel_times), simtel_pe)

        plt.figure()
        plt.plot(t_w, toy_waveform)
        for t in toy_times:
            plt.axvline(t)
        print(len(toy_times), toy_pe)
        plt.show()

        exit()
        viewer = EventViewer(data_stream)
        viewer.draw()

        exit()