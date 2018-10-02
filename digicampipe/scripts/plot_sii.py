"""
plot sii of the time
Usage:
  plot_sii [options] [--] <INPUT>...

Options:
  --help                        Show this
  --plot=FILE                   path of the image to be created, if "none", the
                                plot will be displayed instead of being saveg.
                                [Default: none]
"""
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib import cm
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, subtract_baseline
from digicampipe.utils import DigiCam
from digicampipe.utils.geometry import compute_patch_matrix
from ctapipe.visualization import CameraDisplay

import numpy as np

def entry(files, plot):
    events = calibration_event_stream(files)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)

    patch_matrix = compute_patch_matrix(camera=DigiCam)
    n_patch, n_pixel = patch_matrix.shape
    print(n_patch, n_pixel)

    event = next(events)
    clock_ns = event.data.local_time
    num_samples = len(event.data.adc_samples[0])
    pixel_fadc_sum = np.zeros(shape=(1296, num_samples), dtype=np.float64)
    pixel_fadc_sum += np.array(event.data.adc_samples)
    events_ts = [clock_ns]
    events_id = [event.event_id]

    for i, event in enumerate(events):
        clock_ns = event.data.local_time
        evt_id = event.event_id
        pixel_fadc_sum += np.array(event.data.adc_samples)
        events_ts.append(clock_ns)
        events_id.append(evt_id)
    cnt_evt = float(len(events_ts)+1)
    events_ts = np.array(events_ts)
    events_id = np.array(events_id)

    pixel_fadc_sum /= cnt_evt
    average_wf = np.mean(pixel_fadc_sum, axis=1)
    average_wf_col_vec = average_wf.reshape((1296, 1))
    norm_wf = pixel_fadc_sum - average_wf_col_vec
    #norm_wf_per_patch = patch_matrix.dot(norm_wf)
    #max_corr_noise = np.max(norm_wf_per_patch, axis=1)
    max_corr_noise = np.max(norm_wf, axis=1)
    #max_noise_per_pix = max_corr_noise.dot(patch_matrix)

    fig0 = plt.figure()
    ax = plt.gca()
    display = CameraDisplay(DigiCam.geometry, ax=ax, title='Correlated noise max amplitude')
    display.add_colorbar()
    display.image = max_corr_noise
    plt.show()

    fig1 = plt.figure()
    color_m = cm.get_cmap('viridis')
    plt.subplot(1, 2, 1)
    for i, wf in enumerate(norm_wf_per_patch):
        plt.plot(np.arange(0, num_samples, 1), wf, '-', color=color_m(float(i)/float(432)), linewidth=2)
    plt.subplot(1, 2, 2)
    plt.hist(max_corr_noise, bins=100)

    plt.show()
    '''
    fig1 = plt.figure()
    color_m = cm.get_cmap('viridis')
    # Sector comparison
    plt.subplot(2, 2, 1)
    col_curve = color_m(float(0)/float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[0]/cnt_evt-np.average(pixel_fadc_sum[0]/ cnt_evt)-0.5, '-',
             color=col_curve, label='Pixel 0 (S 1, F 3, Q 11, C 45)', linewidth=2)
    col_curve = color_m(float(432) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[1108] / cnt_evt - np.average(pixel_fadc_sum[1108]/ cnt_evt), '-',
             color=col_curve, label='Pixel 1108 (S 2, F 3, Q 11, C 45)', linewidth=2)
    col_curve = color_m(float(864) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[1038] / cnt_evt - np.average(pixel_fadc_sum[1038]/ cnt_evt)+0.5, '-',
             color=col_curve, label='Pixel 1038 (S 3, F 3, Q 11, C 45)', linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-1.75, 1.75)
    plt.legend(fontsize=10)
    plt.title('FADC Crates', fontsize=14)

    # FADC boards comparison
    plt.subplot(2, 2, 2)
    col_curve = color_m(float(0) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[0] / cnt_evt - np.average(pixel_fadc_sum[0]/ cnt_evt)-0.5, '-',
             color=col_curve, label='Pixel 0 (S 1, F 3, Q 11, C 45)', linewidth=2)
    col_curve = color_m(float(432) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[34] / cnt_evt - np.average(pixel_fadc_sum[34]/ cnt_evt), '-',
             color=col_curve, label='Pixel 34 (S 1, F 6, Q 11, C 45)', linewidth=2)
    col_curve = color_m(float(864) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[104] / cnt_evt - np.average(pixel_fadc_sum[104]/ cnt_evt)+0.5, '-',
             color=col_curve, label='Pixel 104 (S 1, F 2, Q 11, C 45)', linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.ylim(-1.75, 1.75)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.title('FADC boards', fontsize=14)

    # FADC intra-quad comparison
    plt.subplot(2, 2, 3)
    col_curve = color_m(float(0) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[0] / cnt_evt - np.average(pixel_fadc_sum[0]/ cnt_evt)-1., '-',
             color=col_curve, label='Pixel 0 (S 1, F 3, Q 11, C 45)', linewidth=2)
    col_curve = color_m(float(324) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[2] / cnt_evt - np.average(pixel_fadc_sum[2]/ cnt_evt)-0.25, '-',
             color=col_curve, label='Pixel 2 (S 1, F 3, Q 11, C 46)', linewidth=2)
    col_curve = color_m(float(648) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[8] / cnt_evt - np.average(pixel_fadc_sum[8]/ cnt_evt)+0.25, '-',
             color=col_curve, label='Pixel 8 (S 1, F 3, Q 11, C 47)', linewidth=2)
    col_curve = color_m(float(972) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[16] / cnt_evt - np.average(pixel_fadc_sum[16]/ cnt_evt)+1., '-',
             color=col_curve, label='Pixel 16 (S 1, F 3, Q 11, C 48)', linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-2.5, 2.5)
    plt.legend(fontsize=10)
    plt.title('FADC channels', fontsize=14)

    # FADC inter-quad comparison
    plt.subplot(2, 2, 4)
    col_curve = color_m(float(0) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[0] / cnt_evt - np.average(pixel_fadc_sum[0]/ cnt_evt)+1., '-',
             color=col_curve, label='Pixel 0 (S 1, F 3, Q 11, C 45)', linewidth=2)
    col_curve = color_m(float(324) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[3] / cnt_evt - np.average(pixel_fadc_sum[3] / cnt_evt) + 0.25,
             '-', color=col_curve, label='Pixel 3 (S 1, F 3, Q 10, C 41)', linewidth=2)
    col_curve = color_m(float(648) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[4] / cnt_evt - np.average(pixel_fadc_sum[4] / cnt_evt) - 0.25,
             '-', color=col_curve, label='Pixel 4 (S 1, F 3, Q 9, C 39)', linewidth=2)
    col_curve = color_m(float(972) / float(1296))
    plt.plot(np.arange(0, num_samples, 1), pixel_fadc_sum[28] / cnt_evt - np.average(pixel_fadc_sum[28] / cnt_evt) - 1.,
             '-',
             color=col_curve, label='Pixel 28 (S 1, F 3, Q 8, C 33)', linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-2.5, 2.5)
    plt.legend(fontsize=10)
    plt.title('FADC quads', fontsize=14)

    fig2 = plt.figure()
    plt.hist(np.diff(events_ts), np.logspace(2, 7, 250))
    plt.xlabel('$\Delta$t [ns]', fontsize=12)
    plt.ylabel('#events', fontsize=12)

    fig3 = plt.figure()
    plt.plot(np.diff(events_ts), events_ts[:-1])
    plt.xlabel('$\Delta$t [ns]', fontsize=12)
    plt.ylabel('local time [ns]', fontsize=12)

    fig4 = plt.figure()
    plt.plot(events_ts, events_id)
    plt.xlabel('local time [ns]', fontsize=12)
    plt.ylabel('event id', fontsize=12)



    if plot != "none":
        plt.savefig(plot)
    else:
        plt.show()
    plt.close(fig1)
    '''
    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    plot = args['--plot']
    print(files)
    print(plot)
    entry(files, plot)