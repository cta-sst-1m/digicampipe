#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --partition=mono
#SBATCH --mem=16G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='sst-1m-rate-scan'

DIR_NAME=$1 # Directory for the night to produce rate scan
OUTPUT_DIR=$(echo $DIR_NAME | cut -d '/' -f  5,6,7)
OUTPUT_DIR='/sst1m/analyzed/'$OUTPUT_DIR
if ! [[ -d $DIR_NAME ]]; then
    echo "$DIR_NAME must be a directory"
    exit 1
fi

FILES=($(ls -d $DIR_NAME*fits.fz))
# echo ${FILES[@]}



START=0
END=4095
STEP=1
N_SAMPLES=1024


for FILE in "${FILES[@]}"
do
    BASENAME=$(basename "${FILE}")
    BASENAME=$(echo $BASENAME | cut -d'.' -f 1)
    FIGURE=$OUTPUT_DIR'/rate_scan_'$BASENAME'.png'
    OUTPUT=$OUTPUT_DIR'/rate_scan_'$BASENAME'.fits'
    echo digicam-rate-scan --compute --threshold_step=$STEP --threshold_end=$END --threshold_start=$START --n_samples=$N_SAMPLES --figure_path=$FIGURE --output=$OUTPUT $FILE
done



