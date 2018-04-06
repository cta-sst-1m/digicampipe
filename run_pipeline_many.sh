#!/bin/bash

#picture=30
#boundary=5

for picture in 25 30
do
    for boundary in 10 15 20 25 # {5,
    do
    python3 pipeline_crab.py -p ../../sst-1m_data/20171030/ -o output_crab_parametermap_noaddrow_$picture$boundary.npz -s 19 -e 91 -i $picture -b $boundary
    
    done
done

