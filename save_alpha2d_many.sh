#!/bin/bash

picture=30
boundary=10

#for picture in 15 20 25 30
#do
    #for boundary in 15 20 25
    #do
    datafile=../alpha2d_parametermap_addrow/alpha_2d_parametermap_addrow_$picture$boundary.npz
    savefile=../alpha2d_parametermap_addrow/alpha2d_plot_addrow_$picture$boundary.png
    python3 alpha_2d_plot.py -p $datafile -o $savefile
    #convert ../alpha2d_parametermap_noaddrow/alpha2d_plot_noaddrow_$picture$boundary-cropp.png label:$picture$boundary ../alpha2d_parametermap_noaddrow/alpha2d_plot_noaddrow_$picture$boundary-cropp-lab.png
    #done
#done

