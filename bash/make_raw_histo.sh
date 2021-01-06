#!/usr/bin/env bash

FOLDER='/home/alispach/Downloads/dc_nsb_scan/'
numbers=(0 1 2 3 4 5 6 7 8 9 10)

conda activate digicampipe

for i in "${numbers[@]}"
do
    # FILE=$FOLDER'nsb_dc_level_'$i'.hdf5'
    # OUTPUT=$FOLDER'histo_dc_level_'$i'.fits'
    FILE=$FOLDER'nsb_dc_level_template_200MHz'$i'.hdf5'
    OUTPUT=$FOLDER'histo_dc_level_new_pulse'$i'.fits'

    digicam-raw --compute --output=$OUTPUT $FILE

done
