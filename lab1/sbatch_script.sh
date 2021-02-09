#!/bin/bash


#SBATCH -J basicmatmul # Job name
#SBATCH -o logs/basicmatmul.o%j       # Name of stdout output file
#SBATCH -e logs/basicmatmul.e%j       # Name of stderr error file
#SBATCH -p skx-dev # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 48              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=jayeshjo1@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

pwd
date
lscpu

perf list > perf_out.txt
rm -f results/*/*.out  # to get a fresh start
rm -f results/*/*.*.data  # to get a fresh start

# launcher
module load launcher
working_dir=$( realpath . )
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_WORKDIR=$working_dir
export LAUNCHER_JOB_FILE=$LAUNCHER_WORKDIR/launcher_jobs.sh
echo "CHECKING WORKING DIR: $LAUNCHER_JOB_FILE"
$LAUNCHER_DIR/paramrun

python parser.py

# ---------------------------------------------------
