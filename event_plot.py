# Ploting of selected events
############################

# If one wants to plot also Hillas ellipse txt output from pipeline is neccessary.

import numpy as np
import events_image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from optparse import OptionParser


def plot_event(pix_x, pix_y, image, event_no):

    real_event_no = image[event_no, 0]  # the first value in the vector is REAL event number - the save value as saved in the file with Hillas parameters
    image = image[event_no, 1:]
    #print(real_event_no, event_no)
    fig = plt.figure(figsize=(10, 9))
    #ax1 = fig.add_subplot(111)

    plt.scatter(pix_x[image == 0],pix_y[image == 0],color=[0.9, 0.9, 0.9])  # background

    pix_x_event = pix_x[image > 0]
    pix_y_event = pix_y[image > 0]
    image_event = image[image > 0]
    #for i in range(len(pix_x_event)):  # plot event pixels
    #    ax1.plot(pix_x_event[i],pix_y_event[i],'.', color=[image_event[i]/max(image_event), 0, 0])
    plt.scatter(pix_x_event, pix_y_event, c=image_event)
    plt.ylabel('FOV Y [mm]')
    plt.xlabel('FOV X [mm]')  
    plt.colorbar()  


def plot_hillas(hillas, event_no):
    
    
    cen_x = hillas['cen_x'][event_no]
    cen_y = hillas['cen_y'][event_no]
    width = hillas['width'][event_no]
    length = hillas['length'][event_no]
    psi = hillas['psi'][event_no]

    #border_flag = hillas['border'][event_no]
    #print(border_flag)

    ax1 = plt.gca()
    #ax1.plot(cen_x, cen_y, '.') 
    ellipse = Ellipse(xy=[cen_x, cen_y], width=width, height=length, angle=np.rad2deg(psi)+90)
    ax1.add_artist(ellipse)
    ellipse2 = Ellipse(xy=[cen_x, cen_y], width=2*width, height=2*length, angle=np.rad2deg(psi)+90)  # adding larger ellipse
    ax1.add_artist(ellipse2)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--pixels", dest="pixels", help="path to a file with map of pixels", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/pixels.txt')
    parser.add_option("-e", "--events", dest="events", help="path to a file with events", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/events_image_96_proton_ze0_az0.txt')
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_proton_ze0_az0.npz') #False)
    parser.add_option("-n", "--event_number", dest="event_number", help="Number of selected event - N of line in the data files.", default=0)


    (options, args) = parser.parse_args()

    pixels, image = events_image.load_image(options.pixels, options.events)
    pix_x = pixels[0,:]
    pix_y = pixels[1,:]

    event_no = int(options.event_number)

    plot_event(pix_x, pix_y, image, event_no)

    if options.hillas != False:
        hillas = np.load(options.hillas)
        plot_hillas(hillas, event_no)

    plt.show()
