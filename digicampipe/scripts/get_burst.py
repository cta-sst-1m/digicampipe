import numpy as np
import sys
import subprocess



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
  --ouptput=FILE                Path to the output text file listing the bursts
                                and giving the range of timetamps and event_ids
                                for each. If "none", it goes to standard 
                                outpout.
                                [Default: none]
  --expand=INT                  Each burst get expended by the specified number
                                of events
                                [Default: 10]
"""


def expand_mask(input, iters=1):
"""
Expands the True area in an 1D array 'input'.
Expansion occurs by one cell, and is repeated 'iters' times.
"""
xLen, = input.shape
output = input.copy()
for iter in xrange(iters):
  for y in xrange(yLen):
    if (x > 0 and input[x-1]) or (x < xLen - 1 and input[x+1]):
      output[x] = True
  input = output.copy()
return output


args = docopt(__doc__)
files = args['<INPUT>']
event_average = int(args['--event_average'])
threshold_lsb = int(args['--threshold_lsb'])
ouptput = int(args['--ouptput'])
expand = int(args['--expand'])

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
        else:
            bursts[-1].append(event)

# output result
if output == "none":
    run_file = sys.stdout
else:
    run_file = open(output, 'w')
# get info about digicampipe version
digicampipe_branch = subprocess.check_output("cd /home/reniery/ctasoft/digicampipe; git branch | grep \* | cut -d ' ' -f2", shell=True).decode('utf-8').strip('\n')
digicampipe_commit = subprocess.check_output("cd /home/reniery/ctasoft/digicampipe; git rev-parse HEAD", shell=True).decode('utf-8').strip('\n')
# get info about the monitoring pipeline version
pipeline_branch =  subprocess.check_output("cd /home/reniery/cron; git branch | grep \* | cut -d ' ' -f2", shell=True).decode('utf-8').strip('\n')
pipeline_commit = subprocess.check_output("cd /home/reniery/cron; git rev-parse HEAD", shell=True).decode('utf-8').strip('\n')
# write header
run_file.write("#digicampipe branch " + digicampipe_branch+ " commit " + digicampipe_commit + "\n")
run_file.write("#protozfits version " + protozfits.__version__ + '\n')
run_file.write("#monitoring pipeline branch " + pipeline_branch + " commit " + pipeline_commit + "\n")
run_file.write("#burst ts_start ts_end id_start id_end\n")
for i, burst_idxs in enumerate(bursts):
    begin_idx, end_idx = burst_idxs
    run_file.write(i + " " + timestamps[begin_idx] + " " + timestamps[begin_idx])
    run_file.write(" " + event_ids[begin_idx]+  " " + event_ids[end_idx])
    run_file.write("\n")

if output != "none":
    run_file.close()
