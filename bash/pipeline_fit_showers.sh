#!/usr/bin/env bash
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='fit-shower'
#SBATCH --array=0-99

source $HOME/.miniconda3/bin/activate digicampipe

## For Ellipse images
# nsb=('0.0' '0.04' '0.08' '0.2' '0.6')
# nsb=${nsb[$SLURM_ARRAY_TASK_ID]}

# DIRECTORY='/sst1m/MC/digicamtoy/ellipses/'
# INPUT=$DIRECTORY'ellipse_images_'$nsb'GHz.hdf5'
# OUTPUT=$DIRECTORY'ellipse_images_'$nsb'GHz_ter.pk'
# digicam-fit-shower $INPUT --output=$OUTPUT  # --max_events=10 --debug

### For simtel files from Jakub ###
# N_proton files = 2729 (zenith 20)
# N_gamma files = 661 (zenith 20)
# N_gamma diffuse = 570 (zenith 20)

version='v16'
# zenith='zenith_20'
# N_EVENTS=1000
zenith='zenith_20'
# zenith='zenith_20_diffuse'
# particle='proton'
particle='gamma'
DIRECTORY='/sst1m/MC/simtel_krakow_jakub/simtel/'$particle'/'$zenith'/'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/'

ERANGES=('000.1_001.0' '001.0_010.0' '010.0_100.0' '100.0_300.0') # TeV
# ERANGES=('0.5_500.0') # TeV
ERANGE=${ERANGES[$1]}
FILENAMES=$DIRECTORY$particle'_'$ERANGE'TeV_*.simtel.gz'



FILENAMES=( $( ls $FILENAMES ) )
N_FILES=${#FILENAMES[@]}
INDEX=$SLURM_ARRAY_TASK_ID
FILENAME=${FILENAMES[$INDEX]}
RUN_ID=$(ls $FILENAME | grep  -Eo "TeV_[0-9]+" | grep -Eo "[0-9]+")
OUTPUT=$OUTPUT_PATH$zenith'_'$particle'_'$ERANGE'TeV_'$RUN_ID'_'$version'.h5'

# echo $OUTPUT
# rm $OUTPUT
cd ../digicampipe/scripts/
echo "Processing file" $FILENAME "and writing results to" $OUTPUT
python spe_time_lh_fit.py --output=$OUTPUT --debug $FILENAME
echo "Ouput saved to :" $OUTPUT
exit
OUTPUT=$OUTPUT_PATH$zenith'_'$particle'_'$ERANGE'TeV_'$RUN_ID'_trigger.npz'
python read_simtel_all.py $FILENAME --out_path=$OUTPUT
exit