#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out # TO BE CHANGED
#SBATCH --job-name='sst-1m-rate-scan'

source activate digicampipe

START=0
END=4095
STEP=5
N_SAMPLES=1024

DIR_NAME=$1 # Directory for the night to produce rate scan
OUTPUT_DIR=$(echo $DIR_NAME | cut -d '/' -f  5,6,7)
OUTPUT_DIR='/sst1m/analyzed/'$OUTPUT_DIR
if ! [[ -d $DIR_NAME ]]; then
    echo "$DIR_NAME must be a directory"
    exit 1
fi

FILES=($(ls -d $DIR_NAME*fits.fz))

FILES=${FILES[$SLURM_ARRAY_TASK_ID]}
N_FILES=${#FILES[@]}

if ! [[ -z $SLURM_ARRAY_TASK_ID ]]; then

    if [[ $SLURM_ARRAY_TASK_COUNT -ne $N_FILES ]]; then
        echo "Number of array : "$SLURM_ARRAY_TASK_COUNT " must match number of files :"$N_FILES
        exit 1
    fi
fi

for FILE in "${FILES[@]}"
do
    BASENAME=$(basename "${FILE}")
    BASENAME=$(echo $BASENAME | cut -d'.' -f 1)
    FIGURE=$OUTPUT_DIR'/rate_scan_'$BASENAME'.png'
    OUTPUT=$OUTPUT_DIR'/rate_scan_'$BASENAME'.fits'
    digicam-rate-scan --compute --threshold_step=$STEP --threshold_end=$END --threshold_start=$START --n_samples=$N_SAMPLES --figure_path=$FIGURE --output=$OUTPUT $FILE
done

cd $OUTPUT_DIR
convert -delay 1 -loop 0 rate_scan*.png rate_scan.gif