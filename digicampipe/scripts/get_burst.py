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
  --output=FILE                Path to the output text file listing the bursts
                                and giving the range of timetamps and event_ids
                                for each. If "none", it goes to standard 
                                outpout.
                                [Default: none]
  --expand=INT                  Each burst get expended by the specified number
                                of events
                                [Default: 10]
  --merge_sec=FLOAT             Merge bursts if they are closer than the
                                specified amount of seconds
                                [Default: 5]
"""
import numpy as np
import sys
import pandas as pd
from docopt import docopt
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.camera.baseline import fill_digicam_baseline, tag_burst

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


args = docopt(__doc__)
files = args['<INPUT>']
event_average = int(args['--event_average'])
threshold_lsb = float(args['--threshold_lsb'])
output = args['--output']
expand = int(args['--expand'])
merge_sec = float(args['--expand'])

# get events info
events = calibration_event_stream(files)
events = fill_digicam_baseline(events)
events = tag_burst(events, event_average=event_average, 
                   threshold_lsb=threshold_lsb)
n_event = 0
timestamps = []
event_ids = []
are_burst = []
for event in events:
    n_event += 1
    timestamps.append(event.data.local_time)
    event_ids.append(event.event_id)
    are_burst.append(event.data.burst)
timestamps = np.array(timestamps)
event_ids = np.array(event_ids)
are_burst = np.array(are_burst)
if np.all(~are_burst):
    raise SystemExit('no burst detected')

# identify the bursts
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
if merged_bursts[-1][0] != last_burst_begin:
    merged_bursts.append([last_burst_begin, last_burst_end])
print('merged_burst:')
for i, burst in enumerate(merged_bursts):
    beg, end = burst
    print(i, beg, end)
bursts = merged_bursts

# output result
if output == "none":
    run_file = sys.stdout
else:
    run_file = open(output, 'w')
run_file.write("#burst ts_start ts_end id_start id_end\n")  # write header
date_format = '%Y-%m-%dT%H:%M:%S'
for i, burst_idxs in enumerate(bursts):

    ts_begin = pd.to_datetime(timestamps[begin_idx]).strftime(date_format)
    ts_end = pd.to_datetime(timestamps[end_idx]).strftime(date_format)
    run_file.write(str(i) + " " + ts_begin + " " + ts_end)
    run_file.write(" " + str(event_ids[begin_idx]) + " ")
    run_file.write(str(event_ids[end_idx]) + "\n")

if output != "none":
    run_file.close()
