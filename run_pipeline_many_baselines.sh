#!/bin/bash


for baseline1 in 23 24 25 26 27
do
    python3 simtel_pipeline.py -d 1 -e $baseline1 
done

