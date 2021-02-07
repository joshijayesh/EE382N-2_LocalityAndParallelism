#!/bin/bash


#SBATCH -J basicmatmul # Job name
#SBATCH -o logs/basicmatmul.o%j       # Name of stdout output file
#SBATCH -e logs/basicmatmul.e%j       # Name of stderr error file
#SBATCH -p skx-dev # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=jayeshjo1@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

pwd
date
lscpu

perf list > perf_out.txt

rm -rf results  # to get a fresh start

# Launch serial code...

./iter_runs.sh basicmatmul 32 32 32
./iter_runs.sh basicmatmul 512 512 512
./iter_runs.sh basicmatmul 4096 4096 4096

python parser.py

# ---------------------------------------------------
