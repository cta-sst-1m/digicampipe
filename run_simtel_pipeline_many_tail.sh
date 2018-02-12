#!/bin/bash

#picture=30
#boundary=5

for picture in $(seq 21 25)
do
    for boundary in $(seq 0 20)
    do
    python3 simtel_pipeline.py -i $picture -b $boundary
    
    done
done

