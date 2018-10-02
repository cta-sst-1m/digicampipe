"""
plot event_id function of the time
Usage:
  plot_event_id_vs_time [options] [--] <INPUT>...

Options:
  --help                        Show this
  --event_number_min=INT        path to histogram of the dark files
                                [Default: none]
  --event_number_max=INT        Calibration parameters file path
                                [Default: none]
  --plot=FILE                   path of the image to be created, if "none", the
                                plot will be displayed instead of being saveg.
                                [Default: none]
"""
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib import cm
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, subtract_baseline
import numpy as np

def entry(files, event_number_min, event_number_max, plot):
    events = calibration_event_stream(files)
    #events = fill_digicam_baseline(events)
    #events = subtract_baseline(events)
    events_id = []
    events_ts = []
    pixel_wfs = []
    baselines = []
    sample_times = []
    nsamples = []
    for i, event in enumerate(events):
        clock_ns = event.data.local_time
        event_id = event.event_id
        pixel_wf = event.data.adc_samples[1082]
        baseline = event.data.digicam_baseline[1082]
        if event_number_min != "none" and event_id <= int(event_number_min):
            continue
        if event_number_max != "none" and event_id > int(event_number_max):
            continue
        events_ts.append(clock_ns)
        events_id.append(event_id)
        pixel_wfs.append(pixel_wf)
        nsamples.append(len(pixel_wf))
        for i in range(len(pixel_wf)):
            baselines.append(baseline)

    events_ts = np.array(events_ts)
    events_id = np.array(events_id)
    pixel_wfs = np.array(pixel_wfs)
    for i, t in enumerate(events_ts):
        sample_times.append([])
        for j in range(nsamples[i]):
            sample_times[i].append(t+j*4-events_ts[0])


    dead_time = 0
    for i, e in enumerate(events_id):
        if i>0:
            if e-events_id[i-1] > 1:
                dead_time += (e-events_id[i-1]+1)*200e-9

    print('Dead time = ', dead_time*100, '%')
    '''
    fig1 = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(events_ts-events_ts[0], events_id, '.')
    plt.xlabel('$t [ns]$')
    plt.ylabel('event_id')
    #plt.subplot(2, 2, 2)
    #plt.plot(np.diff(events_ts), events_id[1:], '.')
    #plt.xlim(150, 400)
    #plt.xlabel('$\Delta t [ns]$')
    #plt.ylabel('event_id')
    plt.subplot(1, 2, 2)
    plt.hist(np.diff(events_ts), np.arange(150, 400, 4))
    plt.xlim(150, 400)
    plt.xlabel('$\Delta t [ns]$')
    plt.ylabel('# of events')
    #plt.subplot(2, 2, 3)
    #plt.plot(sample_times, pixel_wfs, '.')
    #plt.xlabel('$t [ns]$')
    #plt.ylabel('ADC')
    '''

    fig1 = plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.scatter(events_ts - events_ts[0], events_id, c=(events_id-events_id[0]), marker='o', cmap='viridis')
    #plt.xlabel('$t [ns]$')
    #plt.ylabel('event_id')
    ax1 = fig1.add_subplot(1, 1, 1)
    color_m = cm.get_cmap('viridis')
    allsamples = []
    for i, s in enumerate(sample_times):
        col_curve = color_m(float(i) / len(sample_times))
        if i==0:
            plt.plot(s, pixel_wfs[i], '-', color=col_curve, label='waveform')
        else:
            plt.plot(s, pixel_wfs[i], '-', color=col_curve)
        allsamples.extend(s)
    plt.scatter(allsamples, baselines, marker='o', color='#c10000ff', label='baseline', s=10.0)
    plt.xlabel('$t [ns]$')
    plt.ylabel('ADC')
    plt.legend()
    ax1.set_xticks(range(0, events_ts[-1]-events_ts[0]+201, 200), minor=True)
    plt.grid(which='minor')

    if plot != "none":
        plt.savefig(plot)
    else:
        plt.show()
    plt.close(fig1)
    return


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    event_number_min = args['--event_number_min']
    event_number_max =args['--event_number_max']
    plot = args['--plot']
    print(files)
    print(event_number_min)
    print(plot)
    entry(files, event_number_min, event_number_max, plot)