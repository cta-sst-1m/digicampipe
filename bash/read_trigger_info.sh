#!/usr/bin/env bash
#SBATCH --time=0-00:10:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=3G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='trigger-rate'

### For simtel files from Jakub ###
# N_proton files = 2729 (zenith 20)
# N_gamma files = 661 (zenith 20)
# N_gamma diffuse = 570 (zenith 20)

source $HOME/.miniconda3/bin/activate digicampipe

zenith='20'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/'

GAMMA_FILES='/sst1m/MC/simtel_krakow_jakub/simtel/gamma/zenith_'$zenith'/gamma*TeV*.simtel.gz'
PROTON_FILES='/sst1m/MC/simtel_krakow_jakub/simtel/proton/zenith_'$zenith'/proton*TeV_*.simtel.gz'
GAMMA_DIFFUSE_FILES='/sst1m/MC/simtel_krakow_jakub/simtel/gamma/zenith_'$zenith'_diffuse/gamma*TeV*.simtel.gz'

GAMMA_FILES=( $( ls $GAMMA_FILES) )
echo "N gamma files : "${#GAMMA_FILES[@]}
PROTON_FILES=( $( ls $PROTON_FILES) )
echo "N proton files : "${#PROTON_FILES[@]}
GAMMA_DIFFUSE_FILES=( $( ls $GAMMA_DIFFUSE_FILES) )
echo "N gamma diffuse files : "${#GAMMA_DIFFUSE_FILES[@]}


INDEX=$SLURM_ARRAY_TASK_ID
RUN_ID_1=$(basename -- ${GAMMA_FILES[$INDEX]}) # | grep  -Eo "TeV_[0-9]+" )
RUN_ID_1="${RUN_ID_1%.simtel.gz}"
RUN_ID_2=$(basename -- ${PROTON_FILES[$INDEX]}) # | grep  -Eo "TeV_[0-9]+" )
RUN_ID_2="${RUN_ID_2%.simtel.gz}"
RUN_ID_3=$(basename -- ${GAMMA_DIFFUSE_FILES[$INDEX]}) # | grep  -Eo "TeV_[0-9]+" )
RUN_ID_3="${RUN_ID_3%.simtel.gz}"

cd ../digicampipe/scripts/

OUTPUT=$OUTPUT_PATH'zenith_'$zenith'_'
# python read_simtel_all.py ${GAMMA_FILES[$INDEX]} --out_path=$OUTPUT$RUN_ID_1'.npz'
# python read_simtel_all.py ${PROTON_FILES[$INDEX]} --out_path=$OUTPUT$RUN_ID_2'.npz'
python read_simtel_all.py ${GAMMA_DIFFUSE_FILES[$INDEX]} --out_path=$OUTPUT'diffuse_'$RUN_ID_3'.npz'

exit