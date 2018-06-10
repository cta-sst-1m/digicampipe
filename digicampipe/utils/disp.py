import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from digicampipe.utils import Camera


def disp_eval(parameters, width, length, cog_x, cog_y,
              x_offset, y_offset, psi, skewness,
              size, leakage2, method):

    parameter_values = parameters.valuesdict()

    # (Lessard et al., 2001)
    if method == 1:
        disp_comp = parameter_values['A0']*(1 - width/length)

    # eq 2.10 in Lopez Coto, VHE gamma-ray observations of pulsar
    # wind nebuale ..., but better reference is: (Domingo-Santamaria+, 2005)
    elif method == 2:
        A = (parameter_values['A0'] + parameter_values['A1'] * np.log10(size)
             + pparameter_values['A2'] * np.log10(size)**2)
        B = (parameter_values['A3'] + parameter_values['A4'] * np.log10(size)
             + parameter_values['A5'] * np.log10(size)**2)
        eta = (pparameter_values['A6'] + parameter_values['A7'] * np.log10(size)
               + parameter_values['A8'] * np.log10(size)**2)
        disp_comp = A + B * (width / (length + eta * leakage2))

    # just some test
    elif method == 3:
        disp_comp = parameter_values['A0'] + \
            parameter_values['A1'] * length/width

    # Kranich and Stark, ICRC 2003
    elif method == 4:
        disp_comp = (parameter_values['A0']
                     * (1 - width / (length * (1 + parameter_values['A1'] * leakage2))))
    # (Luke Riley St Marie 2014) <-- simplified Domingo-Santamaria
    elif method == 5:
        fraction = width / (length + parameter_values['A2']
                            * leakage2 * np.log10(size))
        disp_comp = (np.log10(size)
                     * (parameter_values['A0'] + parameter_values['A1'] * (1 - fraction)))

    x_source_comp0 = cog_x + disp_comp*np.cos(psi)  # Two possible solutions
    y_source_comp0 = cog_y + disp_comp*np.sin(psi)  #
    x_source_comp1 = cog_x - disp_comp*np.cos(psi)  #
    y_source_comp1 = cog_y - disp_comp*np.sin(psi)  #

    # Selection of one specific solution according to skewness
    # - head of the shower (close to the source) gives more signal
    # - if skewness > 0, head is closer to the center of FOV than tail
    x_source_comp = np.zeros(disp_comp.shape)
    y_source_comp = np.zeros(disp_comp.shape)

    skewness_is_positive = skewness > 0

    x_source_comp[skewness_is_positive] = x_source_comp1[skewness_is_positive]
    x_source_comp[~skewness_is_positive] = x_source_comp0[~skewness_is_positive]
    y_source_comp[skewness_is_positive] = y_source_comp1[skewness_is_positive]
    y_source_comp[~skewness_is_positive] = y_source_comp0[~skewness_is_positive]

    residuals = np.array([x_offset-x_source_comp, y_offset-y_source_comp])

    # Multi-dimensional minimization needs residuals as
    # a simple 1D vector, thus the .flatten() is applied
    return disp_comp, x_source_comp, y_source_comp, residuals.flatten()


def leak_pixels(pix_x, pix_y, image):

    cam = Camera()
    geom = cam.geometry
    neighbor_matrix = geom.neighbor_matrix
    n_neighbors = np.sum(np.array(neighbor_matrix, dtype=int), axis=0)

    # border pixels
    camera_border_mask = n_neighbors < 6

    # second pixel ring
    n_neighbor_inner = np.sum(
            np.array(np.multiply(neighbor_matrix,
                                 camera_border_mask), dtype=int), axis=1)
    camera_second_ring_mask = n_neighbor_inner >= 3

    # two pixel rings mask
    camera_two_rings_mask = camera_border_mask + camera_second_ring_mask

    # Signal in two outermost pixel rings
    signal_border = np.sum(image[:, camera_two_rings_mask], axis=1)

    # Signal in full image
    signal_full = np.sum(image, axis=1)

    # LEAKAGE2 = the ratio between the light content in the two outermost
    # camera pixel rings and the total light content of the recorded shower
    # image
    leakage2 = signal_border/signal_full

    return leakage2, camera_two_rings_mask, signal_full, signal_border


def arrival_distribution(disp_comp, x_source_comp, y_source_comp, n_triples,
                         theta_squared_cut, bins, x_minmax, y_minmax
                         ):

    # For each event a set of possible arrival directions is calculated
    # as an intersection with another two events, chosen from
    # all events in the dataset. The arrival direction for given set of
    # events is stored if sum of theta^2 for given triplet is less than
    # theta_squared_cut.

    x_intersect = []
    y_intersect = []
    n_bin_values_all = np.zeros((bins, bins))
    theta_squared_sum_hist = []

    for i in trange(len(disp_comp)):

        events1 = np.random.randint(0, len(disp_comp), n_triples)
        events2 = np.random.randint(0, len(disp_comp), n_triples)

        for j, k in zip(events1, events2):

            if j != i and k != j:

                x_triple = [x_source_comp[i],
                            x_source_comp[j],
                            x_source_comp[k]]
                y_triple = [y_source_comp[i],
                            y_source_comp[j],
                            y_source_comp[k]]
                x_mean = np.mean(x_triple)
                y_mean = np.mean(y_triple)

                # Mean arrival direction of the triplet is taken into account
                # only if its 'spread' is not too large. It means that
                # the direction is well defined. As a measure of the spread,
                # sum of theta^2 is taken. Theta means in this case the
                # distance between triplet mean and computed position for each
                # event in the triplet.
                theta_squared_sum = sum(
                    (x_mean-x_triple)**2.0 +
                    (y_mean-y_triple)**2.0
                )

                if theta_squared_sum < theta_squared_cut:
                    x_intersect.append(x_mean)
                    y_intersect.append(y_mean)
                theta_squared_sum_hist.append(theta_squared_sum)

        # binning and normalization
        n_bin = np.histogram2d(x_intersect,
                               y_intersect,
                               bins=bins,
                               range=[x_minmax, y_minmax]
                               )

        if sum(sum(n_bin[0])) > 0:
            n_bin_values = n_bin[0] / sum(sum(n_bin[0]))

            # arrival distribution superposition for all events
            n_bin_values_all = n_bin_values_all + n_bin_values
        x_intersect = []
        y_intersect = []

    return n_bin_values_all, n_bin, theta_squared_sum_hist


# RESOLUTION

def res_gaussian(xy, x0, y0, sigma, H, bkg):     # 2D Gaussian model

    x, y = xy
    theta_squared = (x0-x)**2.0 + (y0-y)**2.0
    G = H * np.exp(-theta_squared/(2*sigma**2.0)) + bkg
    return G


# R68 resolution (if the distribution is gaussian, R68 = sigma)
# Modified so that the events above certain cut are not taken into calculations
# In this version CUT = R99
def r68(x, y, offset_x, offset_y):

    center_x = offset_x     # np.mean(x)
    center_y = offset_y     # np.mean(y)
    x = x - center_x
    y = y - center_y
    N_full = len(x)
    r99 = 0.05
    r68_full = 0.05
    N_in = 0

    while N_in < 0.99*N_full:
        N_in = len(x[(x**2.0 + y**2.0 < r99**2.0)])
        r99 = r99+0.001

    N_in = 0
    while N_in < 0.682*N_full:
        N_in = len(x[(x**2.0 + y**2.0 < r68_full**2.0)])
        r68_full = r68_full+0.001

    cut = r99
    r68 = 0.05
    N_in = 0
    N_full = len(x[(x**2.0 + y**2.0 < cut**2.0)])

    while N_in < 0.682*N_full:
        N_in = len(x[(x**2.0 + y**2.0 < r68**2.0)])
        r68 = r68+0.001

    return r68_full, r68, r99, center_x, center_y


# R68 for modified version of DISP
def r68mod(x, y, n_bin_values, offset_x, offset_y):

    center_x = offset_x     # np.mean(x)
    center_y = offset_y     # np.mean(y)
    x = x - center_x
    y = y - center_y
    N_full = sum(n_bin_values)
    r99 = 0.05
    r68_full = 0.05
    N_in = 0

    while N_in < 0.99*N_full:
        N_in = sum(n_bin_values[(x**2.0 + y**2.0 < r99**2.0)])
        r99 = r99+0.001

    N_in = 0
    while N_in < 0.682*N_full:
        N_in = sum(n_bin_values[(x**2.0 + y**2.0 < r68_full**2.0)])
        r68_full = r68_full+0.001

    cut = r99
    r68 = 0.05
    N_in = 0
    N_full = sum(n_bin_values[(x**2.0 + y**2.0 < cut**2.0)])

    while N_in < 0.682*N_full:
        N_in = sum(n_bin_values[(x**2.0 + y**2.0 < r68**2.0)])
        r68 = r68+0.001

    return r68_full, r68, r99, center_x, center_y


# PLOTTING

def plot_2d(data, vmin, vmax, xlabel, ylabel, cbarlabel):

    rms2 = data[:, 2].reshape(
        (len(np.unique(data[:, 0])),
         len(np.unique(data[:, 1]))
         ))
    x, y = np.meshgrid(np.unique(data[:, 1]), np.unique(data[:, 0]))
    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    plt.imshow(rms2, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label(cbarlabel)
    plt.xticks(range(len(x[0])), x[0])
    plt.yticks(range(len(y[:, 0])), y[:, 0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_event(pix_x, pix_y, image):

    fig = plt.figure(figsize=(9, 9))
    plt.scatter(pix_x[image == 0], pix_y[image == 0], color=[0.9, 0.9, 0.9])
    pix_x_event = pix_x[image > 0]
    pix_y_event = pix_y[image > 0]
    image_event = image[image > 0]
    plt.scatter(pix_x_event, pix_y_event, c=image_event)
    plt.ylabel('FOV Y [mm]')
    plt.xlabel('FOV X [mm]')
    plt.tight_layout()


# function for adding correct ticks corresponding with x,y coordinates to
# the plot instead of indexes of plotted matrix
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]
