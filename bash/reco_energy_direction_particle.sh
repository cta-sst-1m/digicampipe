#!/usr/bin/env bash
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=16G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='train-ml'
#SBATCH --cpus-per-task=12
#SBATCH --array=0-5

source $HOME/.miniconda3/bin/activate digicampipe

version='v15'
zenith='20'
CV=3
# N_EVENTS=200000
GAMMA_CUT=$1
KIND='init'


INPUT_PATH='/sst1m/analyzed/mpeshowerfit/tmp/'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/zenith_'$zenith'/'
mkdir -p $OUTPUT_PATH

proton_file=$INPUT_PATH'/zenith_'$zenith'_proton_'$version'.hdf5'
gamma_file=$INPUT_PATH'/zenith_'$zenith'_gamma_'$version'.hdf5'
gamma_diffuse_file=$INPUT_PATH'/zenith_'$zenith'_diffuse_gamma_'$version'.hdf5'


SIZE_CUTS=('25' '50' '75' '100' '125' '150')
INDEX=$SLURM_ARRAY_TASK_ID
SIZE_CUT=${SIZE_CUTS[$INDEX]}


OUTPUT=$OUTPUT_PATH'size_cut_'$SIZE_CUT'_quater_'$KIND'/'
mkdir -p $OUTPUT
cd ../digicampipe/scripts/
python all_in_one_reco.py --output_directory=$OUTPUT --gamma_diffuse=$gamma_diffuse_file --gamma=$gamma_file --proton=$proton_file --cv=$CV --size_cut=$SIZE_CUT --kind=$KIND --focal=28000
echo "Ouput saved to :" $OUTPUT
exit