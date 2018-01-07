# Ploting of selected events
############################

# If one wants to plot also Hillas ellipse, txt output from pipeline is neccessary.

import numpy as np
import events_image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from optparse import OptionParser


def plot_event(pix_x, pix_y, image, event_no):
    
    real_event_no = image[event_no,0]  # the first value in the vector is REAL event number - the save value as saved in the file with Hillas parameters
    image = image[event_no,1:]
    print(real_event_no, event_no)
    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(111)
    
    ax1.plot(pix_x,pix_y,'.',color=[0.9, 0.9, 0.9])  # background
    
    pix_x_event = pix_x[image > 0]
    pix_y_event = pix_y[image > 0]
    image_event = image[image > 0]
    for i in range(len(pix_x_event)):  # plot event pixels
        ax1.plot(pix_x_event[i],pix_y_event[i],'.', color=[image_event[i]/max(image_event), 0, 0])
    ax1.set_ylabel('FOV Y [mm]')
    ax1.set_xlabel('FOV X [mm]')    


def plot_hillas(hillas, event_no):

    hillas_event = hillas[event_no,:]
    cen_x = hillas_event[1]
    cen_y = hillas_event[2]
    width = hillas_event[4]
    length = hillas_event[3]
    psi = hillas_event[7]

    ax1 = plt.gca()
    #ax1.plot(cen_x, cen_y, '.') 
    ellipse = Ellipse(xy=[cen_x, cen_y], width=width, height=length, angle=np.rad2deg(psi)+90)
    ax1.add_artist(ellipse)

    """
    # if we have npz input. BUT that is useless because the real event number is not unique!
    mask = np.ones(hillas['event_number'].shape[0], dtype=bool)
    mask = hillas['event_number'] == real_event_no

    hillas_cor = dict()
    for key, val in hillas.items():
        hillas_cor[key] = val[mask]
    print(hillas_cor)
    #hillas = hillas[hillas['event_number'] == real_event_no]
    cen_x = hillas_cor['cen_x']
    cen_y = hillas_cor['cen_y']
    """

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--pixels", dest="pixels", help="path to a file with map of pixels", default='pixels.txt')
    parser.add_option("-e", "--events", dest="events", help="path to a file with events", default='events_image.txt')
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a file with hillas parameters", default=False)
    parser.add_option("-n", "--event_number", dest="event_number", help="Number of selected event - N of line in the data files.", default=61)

    (options, args) = parser.parse_args()

    pixels, image = events_image.load_image(options.pixels, options.events)
    pix_x = pixels[0,:]
    pix_y = pixels[1,:]

    event_no = int(options.event_number)

    plot_event(pix_x, pix_y, image, event_no)

    if options.hillas != False:
        hillas = np.loadtxt(options.hillas)
        plot_hillas(hillas, event_no)

    plt.show()
