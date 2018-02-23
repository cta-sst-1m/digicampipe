import numpy as np


def save_hillas_parameters(data_stream, n_showers, output_filename):

    size = []  #np.zeros(n_showers)
    cen_x = []  #np.zeros(n_showers)
    cen_y = []  #np.zeros(n_showers)
    length = []  # np.zeros(n_showers)
    width = []  #np.zeros(n_showers)
    r = []  #np.zeros(n_showers)
    phi = []  #np.zeros(n_showers)
    psi = []  #np.zeros(n_showers)
    miss = []  #np.zeros(n_showers)
    skewness = []  #np.zeros(n_showers)
    kurtosis = []  #np.zeros(n_showers)
    event_number = []  #np.zeros(n_showers)
    time_stamp = []  #np.zeros(n_showers)
    time_spread = []  #np.zeros(n_showers)
    border = []

    for event, i in zip(data_stream, range(n_showers)):

        size.append(event.dl2.shower.size)
        cen_x.append(event.dl2.shower.cen_x.value)
        cen_y.append(event.dl2.shower.cen_y.value)
        length.append(event.dl2.shower.length.value)
        width.append(event.dl2.shower.width.value)
        r.append(event.dl2.shower.r.value)
        phi.append(event.dl2.shower.phi.value)
        psi.append(event.dl2.shower.psi.value)
        miss.append(event.dl2.shower.miss.value)
        skewness.append(event.dl2.shower.skewness)
        kurtosis.append(event.dl2.shower.kurtosis)
        event_number.append(event.r0.event_id)
        time_spread.append(event.dl1.tel[1].time_spread)
        time_stamp.append(event.r0.tel[1].local_camera_clock)
        border.append(1 if event.dl1.tel[1].on_border else 0)

        print('hillas #', i)

    np.savez(
        output_filename,
        size=size,
        cen_x=cen_x,
        cen_y=cen_y,
        length=length,
        width=width,
        r=r,
        phi=phi,
        psi=psi,
        miss=miss,
        skewness=skewness,
        kurtosis=kurtosis,
        event_number=event_number,
        time_stamp=time_stamp,
        time_spread=time_spread,
        border=border
    )


def save_hillas_parameters_in_text(data_stream, output_filename):

    print("Opening output file" + output_filename)
    ofile = open(output_filename, 'w')
    ofile.write(
        "# size cen_x cen_y length width r phi psi miss skewness kurtosis event_number timestamp border time spread\n"
    )
    for event in data_stream:

        size = event.dl2.shower.size
        cen_x = event.dl2.shower.cen_x.value
        cen_y = event.dl2.shower.cen_y.value
        length = event.dl2.shower.length.value
        width = event.dl2.shower.width.value
        r = event.dl2.shower.r.value
        phi = event.dl2.shower.phi.value
        psi = event.dl2.shower.psi.value
        miss = event.dl2.shower.miss.value
        skewness = event.dl2.shower.skewness
        kurtosis = event.dl2.shower.kurtosis
        event_number = event.r0.event_id
        border = 1 if event.dl1.tel[1].on_border else 0
        time_spread = event.dl1.tel[1].time_spread
        time_stamp = event.r0.tel[1].local_camera_clock

        ofile.write(
            str(size) + " " +
            str(cen_x) + " " +
            str(cen_y) + " " +
            str(length) + " " +
            str(width) + " " +
            str(r) + " " +
            str(phi) + " " +
            str(psi) + " " +
            str(miss) + " " +
            str(skewness) + " " +
            str(kurtosis) + " " +
            str(event_number) + " " +
            str(time_stamp) + " " +
            str(border) + " " +
            str(time_spread) + "\n"
        )
        ofile.flush()

    ofile.close()
    print("All done !")
