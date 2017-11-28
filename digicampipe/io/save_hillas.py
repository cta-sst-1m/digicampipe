import numpy as np
from tqdm import tqdm


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


def save_hillas_parameters_in_text(data_stream, output_filename):

    with open(output_filename, 'w') as ofile:

        ofile.write("# size cen_x cen_y length width r phi psi miss skewness kurtosis event_number timestamp border time spread\n")
        for event in tqdm(data_stream):

            size         = event.dl2.shower.size
            cen_x        = event.dl2.shower.cen_x.value
            cen_y        = event.dl2.shower.cen_y.value
            length       = event.dl2.shower.length.value
            width        = event.dl2.shower.width.value
            r            = event.dl2.shower.r.value
            phi          = event.dl2.shower.phi.value
            psi          = event.dl2.shower.psi.value
            miss         = event.dl2.shower.miss.value
            skewness     = event.dl2.shower.skewness
            kurtosis     = event.dl2.shower.kurtosis
            event_number = event.r0.event_id
            border       = 1 if event.dl1.tel[1].on_border else 0
            time_spread  = event.dl1.tel[1].time_spread
            time_stamp = event.r0.tel[1].local_camera_clock

            ofile.write(str(size) + " " + str(cen_x) + " " + str(cen_y) + " " + str(length) + " " + str(width) + " " + str(r) + " "+ str(phi) + " "+ str(psi) + " "+ str(miss) + " "+ str(skewness) + " "+ str(kurtosis) + " "+ str(event_number) + " "+ str(time_stamp) + " " + str(border) + " " + str(time_spread) + "\n")
            ofile.flush()

