#!/usr/bin/env bash

#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out # TO BE CHANGED
#SBATCH --job-name='sst-1m-rate-scan'

conda activate digicampipe

START=0
END=400
STEP=5
N_SAMPLES=1024

OUTPUT_DIR='/home/alispach/Downloads/test_simtel/'

FILES=(nsb_only_0.hdf5 nsb_only_1.hdf5 nsb_only_2.hdf5 nsb_gain_drop_0.hdf5 nsb_gain_drop_1.hdf5 nsb_gain_drop_2.hdf5 nsb_voltage_drop_0.hdf5 nsb_voltage_drop_1.hdf5 nsb_voltage_drop_2.hdf5)
N_FILES=${#FILES[@]}

i=0
for FILE in ${FILES[@]}
do
    OUTPUT_1=$OUTPUT_DIR'rate_scan_'$i'.fits'
    OUTPUT_2=$OUTPUT_DIR'raw_histo_'$i'.fits'
    FILE=$OUTPUT_DIR$FILE
    i=$((i+1))
    digicam-raw --compute --output=$OUTPUT_2 $FILE
    digicam-rate-scan --compute --threshold_step=$STEP --threshold_end=$END --threshold_start=$START --n_samples=$N_SAMPLES --output=$OUTPUT_1 $FILE
done