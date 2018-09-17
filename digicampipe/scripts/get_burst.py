"""
Get a list of bursts
Usage:
  digicam-get-burst [options] [--] <INPUT>...

Options:
  --help                        Show this
  --event_average=INT           Number of events on which the moving average 
                                of the baselines are calculated
                                [Default: 100]
  --threshold_lsb=FLOAT         How much the baseline must be above the moving
                                average for the event to be taken as part of a 
                                burst
                                [Default: 2.0]
  --output=FILE                 Path to the output text file listing the bursts
                                and giving the range of timetamps and event_ids
                                for each. If "none", it goes to standard 
                                outpout.
                                [Default: none]
  --expand=INT                  Each burst get expended by the specified number
                                of events
                                [Default: 10]
  --plot_baseline=FILE          Path to the outpout plot of the histroy of the 
                                mean baseline. Set to "none" to not make that 
                                plot or to "show" to display it.
                                [Default: show]
  --merge_sec=FLOAT             Merge bursts if they are closer than the
                                specified amount of seconds
                                [Default: 5.0]
  --video_prefix=FILE           Prefix of the output video files. One video per
                                burst (with path=prefix + "_" + str(burst_idx)
                                + ".mp4")
                                The videos show the baseline evolution during
                                a burst. Set to "none" to not make any video or
                                set to "show" to display them.
                                [Default: none]

"""
import numpy as np
import sys
import pandas as pd
from docopt import docopt
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, tag_burst
from matplotlib import pyplot as plt
from pandas import to_datetime
from ctapipe.visualization import CameraDisplay
from digicampipe.utils import DigiCam
import matplotlib.animation as animation


def expand_mask(input, iters=1):
    """
    Expands the True area in an 1D array 'input'.
    Expansion occurs by one cell, and is repeated 'iters' times.
    """
    xLen, = input.shape
    output = input.copy()
    for iter in range(iters):
        for x in range(xLen):
            if (x > 0 and input[x-1]) or (x < xLen - 1 and input[x+1]):
                output[x] = True
        input = output.copy()
    return output


def animate_baseline(events, video, event_id_min=None, event_id_max=None):
    fig = plt.figure()
    ax = plt.gca()
    display = CameraDisplay(DigiCam.geometry, ax=ax)
    display.add_colorbar()
    title = plt.title("")
    event_ids = []
    ts = []
    baselines = []
    for event in events:
        if event_id_min and event.event_id <= event_id_min:
            continue
        if event_id_max and event.event_id >= event_id_max:
            break
        event_ids.append(event.event_id)
        ts.append(event.data.local_time)
        baselines.append(event.data.digicam_baseline)
    t0 = ts[0]
    title_text = 'Baseline evolution\n' + 'event: %i, t=%.3f s' % \
                 (event_ids[0], np.round((ts[0] - t0) * 1e-9, 3))
    title.set_text(title_text)
    display.image = baselines[0]

    def update(i):
        event_id = event_ids[i]
        t = ts[i]
        title_text = 'Baseline evolution\n' + 'event: %i, t=%.3f s' \
                     %(event_id, np.round((t - t0) * 1e-9, 3))
        title.set_text(title_text)
        display.image = baselines[i]
        plt.pause(.04)
        return display, title

    nframe = len(ts)
    print('creating animation with',nframe , 'frames')
    anim = animation.FuncAnimation(fig, update, save_count=nframe + 1,
                                   frames=np.arange(1, nframe),
                                   interval=100)
    print('saving...')
    plt.rcParams['animation.ffmpeg_path'] = \
        u'/home/yves/anaconda3/envs/digicampipe/bin/ffmpeg'
    if video != "show":
        # import logging
        # logger = logging.getLogger('matplotlib.animation')
        # logger.setLevel(logging.DEBUG)
        anim.save(video, writer='ffmpeg',
                  metadata=dict(artist='yves.renier@unige.ch',
                                comment='baseline during burst'),
                  codec='mpeg4', bitrate=20000)
        print('video saved as', video)
    else:
        plt.show()
    plt.close(fig)


def entry(files, plot_baseline="show", event_average=100, threshold_lsb=2., 
          output="none", expand=10, merge_sec=5., video_prefix="none"):
    # get events info
    events = calibration_event_stream(files)
    events = fill_digicam_baseline(events)
    events = tag_burst(events, event_average=event_average, 
                       threshold_lsb=threshold_lsb)
    n_event = 0
    timestamps = []
    event_ids = []
    are_burst = []
    baselines = []
    for event in events:
        n_event += 1
        timestamps.append(event.data.local_time)
        event_ids.append(event.event_id)
        are_burst.append(event.data.burst)
        baselines.append(np.mean(event.data.digicam_baseline))
    timestamps = np.array(timestamps)
    event_ids = np.array(event_ids)
    are_burst = np.array(are_burst)
    baselines = np.array(baselines)
    
    # plot history of the baselines
    if plot_baseline.lower() != "none":
        fig1 = plt.figure(figsize=(8,6))
        plt.plot_date(to_datetime(timestamps), baselines, '.')
        plt.ylabel('mean baseline [LSB]')
        if plot_baseline.lower() == "show":
            plt.show()
        else:
            plt.savefig(plot_baseline)
        plt.close(fig1)

    # identify the bursts
    if np.all(~are_burst):
        raise SystemExit('no burst detected')
    are_burst = expand_mask(are_burst, iters=expand)
    previous_is_burst = False
    bursts = []
    for event in range(n_event):
        if are_burst[event]:
            if not previous_is_burst:
                bursts.append([event])
            if event == n_event - 1 or (not are_burst[event + 1]):
                bursts[-1].append(event)
            previous_is_burst = True
        else:
            previous_is_burst = False
    if np.all(~are_burst):
        raise SystemExit('no burst identified')

    # merge bursts which are closer than merge_sec seconds
    last_burst_begin = bursts[0][0]
    last_burst_end = bursts[0][1]
    merged_bursts = []
    n_burst = len(bursts)
    for burst_idxs in bursts[1:]:
        begin_idx, end_idx = burst_idxs
        if (timestamps[begin_idx] - timestamps[last_burst_end]) < merge_sec * 1e9:
            last_burst_end = end_idx
        else:
            merged_bursts.append([last_burst_begin, last_burst_end])
            last_burst_begin = begin_idx
            last_burst_end = end_idx
    if len(merged_bursts) == 0 or merged_bursts[-1][0] != last_burst_begin:
        merged_bursts.append([last_burst_begin, last_burst_end])
    bursts = merged_bursts

    # output result
    if output == "none":
        run_file = sys.stdout
    else:
        run_file = open(output, 'w')
    run_file.write("#burst ts_start ts_end id_start id_end\n")  # write header
    date_format = '%Y-%m-%dT%H:%M:%S'
    for i, burst_idxs in enumerate(bursts):
        begin_idx, end_idx = burst_idxs
        ts_begin = pd.to_datetime(timestamps[begin_idx]).strftime(date_format)
        ts_end = pd.to_datetime(timestamps[end_idx]).strftime(date_format)
        run_file.write(str(i) + " " + ts_begin + " " + ts_end)
        run_file.write(" " + str(event_ids[begin_idx]) + " ")
        run_file.write(str(event_ids[end_idx]) + "\n")
        if video_prefix != "none":
            first_event_id = event_ids[begin_idx]
            events = calibration_event_stream(files, event_id=first_event_id)
            events = fill_digicam_baseline(events)
            if video_prefix != "show":
                video = video_prefix + "_" + str(i) + ".mp4"
            else:
                video = "show"
            animate_baseline(events, video, event_id_min=event_ids[begin_idx],
                             event_id_max=event_ids[end_idx])
    if output != "none":
        run_file.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    files = args['<INPUT>']
    event_average = int(args['--event_average'])
    threshold_lsb = float(args['--threshold_lsb'])
    output = args['--output']
    expand = int(args['--expand'])
    merge_sec = float(args['--merge_sec'])
    plot_baseline = args['--plot_baseline']
    video_prefix = args['--video_prefix']
    entry(files, plot_baseline, event_average, threshold_lsb, output, expand, 
          merge_sec, video_prefix)

