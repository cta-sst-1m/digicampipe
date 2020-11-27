#!/usr/bin/env bash
#SBATCH --time=00-12:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='fit-shower'
#SBATCH --array=1-10

source $HOME/.miniconda3/bin/activate digicampipe

## For Ellipse images
NSB=('0.0' '0.04' '0.08' '0.2' '0.6')
NSB=${NSB[$SLURM_ARRAY_TASK_ID]}
NSB='0.0'

DIRECTORY='/sst1m/MC/digicamtoy/ellipses/'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/'
VERSION='v12'
ID=$SLURM_ARRAY_TASK_ID
# ID='1'
FILENAME=$DIRECTORY'ellipse_images_'$NSB'GHz_'$VERSION'_id_'$ID'.hdf5'

OUTPUT=$OUTPUT_PATH'ellipse_images_'$NSB'GHz_'$VERSION'_id_'$ID'_bis.pk'

# echo $OUTPUT
# rm $OUTPUT
echo "Processing file" $FILENAME "and writing results to" $OUTPUT
digicam-fit-shower-final --output=$OUTPUT $FILENAME
# digicam-fit-shower-final -v --output=$OUTPUT $FILENAME
# ls $FILENAME | wc -l
echo "Ouput saved to :" $OUTPUT
exit