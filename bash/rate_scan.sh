#!/usr/bin/env bash

FILE=$1
START=0
END=4095
STEP=1
N_SAMPLES=1024

DIR=$(dirname "${FILE}")
DIR=$(echo $DIR | cut -d '/' -f  5,6,7)
DIR='/sst1m/analyzed/'$DIR
ls $DIR -ltr

BASENAME=$(basename "${FILE}")
BASENAME=$(echo $BASENAME | cut -d'.' -f 1)
FIGURE=$DIR'/rate_scan_'$BASENAME'.png'
OUTPUT=$DIR'/rate_scan_'$BASENAME'.fits'

echo digicam-rate-scan --compute --threshold_step=$STEP --threshold_end=$END --threshold_start=$START --n_samples=$N_SAMPLES --figure_path=$FIGURE --output=$OUTPUT $FILE