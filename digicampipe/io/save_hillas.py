import numpy as np


def save_hillas_parameters(data_stream, n_showers, output_filename):

    size = np.zeros(n_showers)
    cen_x = np.zeros(n_showers)
    cen_y = np.zeros(n_showers)
    length = np.zeros(n_showers)
    width = np.zeros(n_showers)
    r = np.zeros(n_showers)
    phi = np.zeros(n_showers)
    psi = np.zeros(n_showers)
    miss = np.zeros(n_showers)
    skewness = np.zeros(n_showers)
    kurtosis = np.zeros(n_showers)
    event_number = np.zeros(n_showers)
    time_stamp = np.zeros(n_showers)

    for event, i in zip(data_stream, range(n_showers)):

        size[i] = event.dl2.shower.size
        cen_x[i] = event.dl2.shower.cen_x.value
        cen_y[i] = event.dl2.shower.cen_y.value
        length[i] = event.dl2.shower.length.value
        width[i] = event.dl2.shower.width.value
        r[i] = event.dl2.shower.r.value
        phi[i] = event.dl2.shower.phi.value
        psi[i] = event.dl2.shower.psi.value
        miss[i] = event.dl2.shower.miss.value
        skewness[i] = event.dl2.shower.skewness
        kurtosis[i] = event.dl2.shower.kurtosis
        event_number[i] = event.r0.event_id

        for telescope_id in event.r0.tels_with_data:

            time_stamp[i] = event.r0.tel[telescope_id].local_camera_clock

        print('hillas #', i)

    np.savez(output_filename, size=size, cen_x=cen_x, cen_y=cen_y, length=length, width=width, r=r, phi=phi, psi=psi,
             miss=miss, skewness=skewness, kurtosis=kurtosis, event_number=event_number, time_stamp=time_stamp)
    
    
def save_hillas_parameters_in_text(data_stream, output_filename):

    size         = 0
    cen_x        = 0
    cen_y        = 0
    length       = 0
    width        = 0
    r            = 0
    phi          = 0
    psi          = 0
    miss         = 0
    skewness     = 0
    kurtosis     = 0
    event_number = 0
    time_stamp   = 0
    
    num_showers = 0
    print("Opening output file" + output_filename)
    ofile = open(output_filename, 'w')
    ofile.write("# size cen_x cen_y length width r phi psi miss skewness kurtosis event_number timestamp border\n")
    for event in data_stream:

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
        border       = event.dl1.tel[1].on_border
        
        on_border = 0
        if border == True:
            on_border = 1

        for telescope_id in event.r0.tels_with_data:
            time_stamp = event.r0.tel[telescope_id].local_camera_clock
        ofile.write(str(size) + " " + str(cen_x) + " " + str(cen_y) + " " + str(length) + " " + str(width) + " " + str(r) + " "+ str(phi) + " "+ str(psi) + " "+ str(miss) + " "+ str(skewness) + " "+ str(kurtosis) + " "+ str(event_number) + " "+ str(time_stamp) + " " + str(on_border) + "\n")
        ofile.flush()
        num_showers = num_showers + 1
        #if num_showers >= 10:
        #    break
        
    ofile.close()
    print("All done !")
    
