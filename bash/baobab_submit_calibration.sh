#!/bin/env bash

echo -e "Starting job: " `date` '\nJob array ID ' ${SLURM_ARRAY_TASK_ID}

# sbatch 1_raw_ghv_off.sh
# sbatch 2_raw_ghv_on.sh
# sbatch 3_raw_dark.sh
sbatch 4_spe.sh
job_id_1=$(sbatch 5_timing.sh)
job_id_2=$(sbatch 6_fmpe.sh --dependency=afterok:$job_id_1)
sbatch 7_mpe.sh --dependency=afterok:$job_id_2
sbatch 8_baseline_shift.sh

echo "Ending job: " `date`
