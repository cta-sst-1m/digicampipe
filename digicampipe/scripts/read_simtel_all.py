#!/usr/bin/env python
"""
Read all events in simtel.gz files and save them as .npz file

Usage:
  ./read_simtel_all.py <FILES>... [options]

Options:
  -o PATH --out_path=PATH     Path to store the output data [Default: ./output]
  -i N    --max_id=N          Max event id to be saved. Once that an event has an higher id that N, the reading of the simtel file is stopped. If none the file is read to the end. [Default: none]
"""

from eventio.simtel.simtelfile import SimTelFile
from docopt import docopt
import numpy as np
from tqdm import tqdm


def save_file(
        filename_showerparam, event_id, energy, x_core, y_core, h_first_int, 
        theta, phi, tel_pos_all, n_trig_list
):
    np.savez(
        filename_showerparam, event_id=event_id, energy=energy, x_core=x_core, 
        y_core=y_core, h_first_int=h_first_int, tel_pos_all=tel_pos_all, theta=theta, 
        phi=phi, n_trig=n_trig_list
    )
    print(filename_showerparam, 'has been saved.')

def entry(files, output, max_event_id=None):
    event_id_all = []
    energy = []
    x_core = []
    y_core = []
    h_first_int = []
    theta = []
    phi = []
    tel_triggered_all = []
    all_tel_pos = None
    for file in tqdm(files):
        event_id = []
        tel_triggered = []
        with SimTelFile(file) as f:
            all_tel_pos = f.header['tel_pos']
            mc_iter = f.iter_mc_events()
            try:
                for event in tqdm(mc_iter):
                    evt_id = event['event_id']
                    if max_event_id is not None and evt_id > max_event_id:
                        print('max event_id reached for file', file)
                        break
                    mc_shower = event['mc_shower']
                    mc_event = event['mc_event']
                    event_id.append(evt_id)
                    energy.append(mc_shower['energy'])
                    x_core.append(mc_event['xcore'])
                    y_core.append(mc_event['ycore'])
                    h_first_int.append(mc_shower['h_first_int'])
                    theta.append(90 - np.rad2deg(mc_shower['altitude']))
                    phi.append(np.rad2deg(mc_shower['azimuth']))
                    tel_triggered.append([])
            except:
                print("WARNING: unexpected end of file", file)
                pass
        with SimTelFile(file) as f:
            try:
                trig_iter = f.iter_array_events()
                for event in trig_iter:
                    evt_id = event['event_id']
                    if max_event_id is not None and evt_id >= max_event_id:
                        print('max event_id reached for file', file)
                        break
                    mc_event_index = event_id.index(evt_id)
                    tel_triggered[mc_event_index] = event['trigger_information']['triggered_telescopes']
            except:
                print("WARNING: unexpected end of file", file)
                pass
        event_id_all.extend(event_id)
        tel_triggered_all.extend(tel_triggered)
    nevent = len(event_id_all)
    ntrig_list = []
    for t in tel_triggered_all:
        ntrig_list.append(len(t))
    ntrig = np.sum(np.array(ntrig_list) >=1 )
    print(ntrig, '/', nevent, 'event triggered')
    save_file(
        output, event_id_all, energy, x_core, y_core, h_first_int,
        theta, phi, all_tel_pos, ntrig_list,
    )
          

if __name__ == '__main__':

    args = docopt(__doc__)
    files = args["<FILES>"]
    output = args['--out_path']
    max_event_id = args['-i']
    if max_event_id.lower() == "none":
        max_event_id = None
    else:
        max_event_id = int(max_event_id)
    entry(files, output, max_event_id)

