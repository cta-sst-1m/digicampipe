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
        print('hillas #', i)

    np.savez(output_filename, size=size, cen_x=cen_x, cen_y=cen_y, length=length, width=width, r=r, phi=phi, psi=psi,
             miss=miss, skewness=skewness, kurtosis=kurtosis)