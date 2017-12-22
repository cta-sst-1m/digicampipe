
import numpy as np
import sys
import matplotlib.pyplot as plt
from alpha_2d_plot import plot_alpha2d


def plot_aperture(x0, y0, r, col):
    
    phi = np.linspace(0,2*np.pi,1000)
    x = r*np.cos(phi) + x0
    y = r*np.sin(phi) + y0
    ax1 = plt.gca()
    ax1.plot(x,y,'-',color=col)


def plot_onoffcirc(x0, y0):
    
    phi = np.linspace(0,2*np.pi,1000)
    r = np.sqrt(x0**2.0 + y0**2.0)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    ax1 = plt.gca()
    ax1.plot(x,y,'k--')

	
def count_aperture(data, x0, y0, r):

    N = data['N'][np.sqrt((data['x']-x0)**2 + (data['y']-y0)**2) <= r]
    N = sum(N)
    return N
	

def off_aperture1(x0, y0):
    return -x0,-y0
    
	
def off_aperture3(x0, y0):
    x = np.array([-x0,-y0,y0])
    y = np.array([-y0,x0,-x0])
    return x,y


def lima_significance(N_on, N_off, alpha): # Li and Ma, 1983
    
    sign_lima = np.sqrt( 2.0*( N_on*np.log( ((1.0+alpha)/alpha) * (N_on/(N_on+N_off)) ) + N_off*np.log( (1.0+alpha) * (N_off/(N_on+N_off)) ) ) )
    return sign_lima
	

if __name__ == '__main__':

    x0 = 20
    y0 = 43
    r = 30

    data = np.load(sys.argv[1])

    x_off,y_off = off_aperture1(x0,y0)
    x_off3,y_off3 = off_aperture3(x0,y0)

    N_on = count_aperture(data,x0,y0,r)
    N_off = count_aperture(data,x_off,x_off,r) #one N_off region

    N_off3 = 0
    for i in range(3):
        N_off3 = N_off3 + count_aperture(data,x_off3[i],y_off3[i],r) #multiple N_off regions with respect to the image center

    alpha = 1.0/3.0 #alpha = 1/number of OFF regions
    sig = lima_significance(N_on,N_off3,alpha)
    #sig = lima_significance(N_on,N_off,1.0)
    print(sig)

    plot_alpha2d(data)
    plot_aperture(x0,y0,r,'black')
    plot_onoffcirc(x0,y0)
    #plot_aperture(x_off,y_off,r)
    for i in range(3):
        plot_aperture(x_off3[i],y_off3[i],r,'magenta')

    plt.show()
    
