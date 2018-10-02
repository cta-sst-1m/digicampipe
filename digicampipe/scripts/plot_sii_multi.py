"""
plot waveform sum for different runs
Usage:
  plot_sii_multi [options] [--] ...

Options:
  --help                        Show this
  --run1=FILE                   path of the first set of runs to be analysed
  --run2=FILE                   path of the second set of runs to be analysed
"""
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib import cm
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, subtract_baseline
import numpy as np


def entry(files_1, files_2):
    events = []
    events.append(calibration_event_stream(files_1))
    events.append(calibration_event_stream(files_2))

    pix_wf_sum = []
    num_samp = []
    cnt_evt = []
    for i, event in enumerate(events):
        event = fill_digicam_baseline(event)
        event = subtract_baseline(event)
        evt = next(event)
        clock_ns = evt.data.local_time
        num_samp.append(len(evt.data.adc_samples[0]))
        pixel_fadc_sum = np.zeros(shape=(1296, len(evt.data.adc_samples[0])), dtype=np.float64)
        pixel_fadc_sum += np.array(evt.data.adc_samples)
        print(pixel_fadc_sum.shape, num_samp[-1])
        events_ts = [clock_ns]
        for evt in event:
            clock_ns = evt.data.local_time
            pixel_fadc_sum += np.array(evt.data.adc_samples)
            events_ts.append(clock_ns)
        cnt_evt.append(float(len(events_ts)+1))
        pix_wf_sum.append(pixel_fadc_sum)

    leg_stat = ['HV ON: ', 'HV OFF: ']
    line_stat = ['-', '--']
    fig1 = plt.figure()
    color_m = cm.get_cmap('viridis')
    # Sector comparison
    plt.subplot(2, 2, 1)
    for i, cnt in enumerate(cnt_evt):
        col_curve = color_m(float(0)/float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][0]/cnt-np.average(pix_wf_sum[i][0]/cnt)-0.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 0 (S 1, F 3, Q 11, C 45)',
                 linewidth=2)
        col_curve = color_m(float(432)/float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][1108]/cnt-np.average(pix_wf_sum[i][1108]/cnt),
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 1108 (S 2, F 3, Q 11, C 45)',
                 linewidth=2)
        col_curve = color_m(float(864)/float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][1038]/cnt-np.average(pix_wf_sum[i][1038]/cnt)+0.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 1038 (S 3, F 3, Q 11, C 45)',
                 linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-1.75, 1.75)
    plt.legend(fontsize=10)
    plt.title('FADC Crates', fontsize=14)

    # FADC boards comparison
    plt.subplot(2, 2, 2)
    for i, cnt in enumerate(cnt_evt):
        col_curve = color_m(float(0) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][0]/cnt-np.average(pix_wf_sum[i][0]/cnt, axis=0)-0.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 0 (S 1, F 3, Q 11, C 45)',
                 linewidth=2)
        col_curve = color_m(float(432) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][34]/cnt-np.average(pix_wf_sum[i][34]/cnt),
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 34 (S 1, F 6, Q 11, C 45)',
                 linewidth=2)
        col_curve = color_m(float(864) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][104]/cnt-np.average(pix_wf_sum[i][104]/cnt)+0.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 104 (S 1, F 2, Q 11, C 45)',
                 linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.ylim(-1.75, 1.75)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.title('FADC boards', fontsize=14)

    # FADC intra-quad comparison
    plt.subplot(2, 2, 3)
    for i, cnt in enumerate(cnt_evt):
        col_curve = color_m(float(0) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][0]/cnt-np.average(pix_wf_sum[i][0]/cnt)-1.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 0 (S 1, F 3, Q 11, C 45)',
                 linewidth=2)
        col_curve = color_m(float(324) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][2]/cnt-np.average(pix_wf_sum[i][2]/cnt)-0.5,
                 linestyle=line_stat[i],color=col_curve, label=leg_stat[i]+'Pixel 2 (S 1, F 3, Q 11, C 46)',
                 linewidth=2)
        col_curve = color_m(float(648) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][8]/cnt-np.average(pix_wf_sum[i][8]/cnt)+0.5,
                 linestyle=line_stat[i],color=col_curve, label=leg_stat[i]+'Pixel 8 (S 1, F 3, Q 11, C 47)',
                 linewidth=2)
        col_curve = color_m(float(972) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][16]/cnt-np.average(pix_wf_sum[i][16]/cnt)+1.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 16 (S 1, F 3, Q 11, C 48)',
                 linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-2.5, 2.5)
    plt.legend(fontsize=10)
    plt.title('FADC channels', fontsize=14)

    # FADC inter-quad comparison
    plt.subplot(2, 2, 4)
    for i, cnt in enumerate(cnt_evt):
        col_curve = color_m(float(0) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][0]/cnt-np.average(pix_wf_sum[i][0]/cnt)-1.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 0 (S 1, F 3, Q 11, C 45)',
                 linewidth=2)
        col_curve = color_m(float(324) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][3]/cnt-np.average(pix_wf_sum[i][3]/cnt)-0.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 3 (S 1, F 3, Q 10, C 41)',
                 linewidth=2)
        col_curve = color_m(float(648) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][4]/cnt-np.average(pix_wf_sum[i][4]/cnt)+0.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 4 (S 1, F 3, Q 9, C 39)',
                 linewidth=2)
        col_curve = color_m(float(972) / float(1296))
        plt.plot(np.arange(0, num_samp[i]), pix_wf_sum[i][28]/cnt-np.average(pix_wf_sum[i][28]/cnt)+1.5,
                 linestyle=line_stat[i], color=col_curve, label=leg_stat[i]+'Pixel 28 (S 1, F 3, Q 8, C 33)',
                 linewidth=2)
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('ADC sum / #events', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-2.5, 2.5)
    plt.legend(fontsize=10)
    plt.title('FADC quads', fontsize=14)

    plt.show()
    #plt.close(fig1)

    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files_1 = args['--run1']
    files_2 = args['--run2']
    entry(files_1, files_2)