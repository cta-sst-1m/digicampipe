# Ploting of selected events

import numpy as np
import events_image
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    
    pixels, image = events_image.load_image('pixels.txt', 'events_image.txt')
    pix_x = pixels[0,:]
    pix_y = pixels[1,:]
    
    event_no = 21  # specific event selection. The number is not REAL event number but number of saved event (N of a line with event)
    
    plot_event(pix_x, pix_y, image, event_no)
    plt.show()
