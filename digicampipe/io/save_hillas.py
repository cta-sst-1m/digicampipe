import numpy as np
from tqdm import tqdm
from collections import OrderedDict

what_to_write_description = {
    # output name:  how-to-get-the-value-from-the-event
    'size': lambda e: e.dl2.shower.size,
    'cen_x': lambda e: e.dl2.shower.cen_x.value,
    'cen_y': lambda e: e.dl2.shower.cen_y.value,
    'length': lambda e: e.dl2.shower.length.value,
    'width': lambda e: e.dl2.shower.width.value,
    'r': lambda e: e.dl2.shower.r.value,
    'phi': lambda e: e.dl2.shower.phi.value,
    'psi': lambda e: e.dl2.shower.psi.value,
    'miss': lambda e: e.dl2.shower.miss.value,
    'skewness': lambda e: e.dl2.shower.skewness,
    'kurtosis': lambda e: e.dl2.shower.kurtosis,
    'event_number': lambda e: e.r0.event_id,
    'border': lambda e: 1 if e.dl1.tel[1].on_border else 0,
    'time_spread': lambda e: e.dl1.tel[1].time_spread,
    'time_stamp': lambda e: e.r0.tel[1].local_camera_clock,
}

def save_hillas_parameters(data_stream, output_filename):

    output = {
        'size': [],
        'cen_x': [],
        'cen_y': [],
        'length': [],
        'width': [],
        'r': [],
        'phi': [],
        'psi': [],
        'miss': [],
        'skewness': [],
        'kurtosis': [],
        'event_number': [],
        'time_stamp': [],
        'time_spread': [],
    }

    for i, event in enumerate(tqdm(data_stream)):

        output['size'].append(event.dl2.shower.size)
        output['cen_x'].append(event.dl2.shower.cen_x.value)
        output['cen_y'].append(event.dl2.shower.cen_y.value)
        output['length'].append(event.dl2.shower.length.value)
        output['width'].append(event.dl2.shower.width.value)
        output['r'].append(event.dl2.shower.r.value)
        output['phi'].append(event.dl2.shower.phi.value)
        output['psi'].append(event.dl2.shower.psi.value)
        output['miss'].append(event.dl2.shower.miss.value)
        output['skewness'].append(event.dl2.shower.skewness)
        output['kurtosis'].append(event.dl2.shower.kurtosis)
        output['event_number'].append(event.r0.event_id)
        output['time_spread'].append(event.dl1.tel[1].time_spread)
        output['time_stamp'].append(event.r0.tel[1].local_camera_clock)

    np.savez(output_filename, **output)


def save_hillas_parameters_in_text(
    data_stream,
    output_filename,
    description=None
):
    description = description or what_to_write_description
    with open(output_filename, 'w') as ofile:
        ofile.write("# {keys}\n".format(' '.join(description)))
        for event in tqdm(data_stream):
            for get_value_from in description.values():
                ofile.write(str(get_value_from(event)) + " ")
            ofile.write('\n')
            ofile.flush()
