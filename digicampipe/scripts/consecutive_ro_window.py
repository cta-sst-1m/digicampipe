'''
Stacking readout windows to illustrate bursts
Usage:
  consecutive_ro_window [options] [--] <INPUT>...

Options:
  -h --help     Show this screen.
  -s                          first event_id
  -e                          last event_id
  -o OUTPUT --output=OUTPUT   Folder where to store the results.
  -c --compute                Compute the data.
  -f --fit                    Fit
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --dc_levels=<DAC>           LED DC DAC level
  --save_figures              Save the plots to the OUTPUT folder
  --gain=<GAIN_RESULTS>       Calibration params to use in the fit
  --template=<TEMPLATE>       Templates measured
  --crosstalk=<CROSSTALK>     Calibration params to use in the fit
'''
from digicampipe.io.event_stream import event_stream
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from tqdm import tqdm
