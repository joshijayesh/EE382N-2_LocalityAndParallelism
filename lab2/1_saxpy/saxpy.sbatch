#!/bin/bash
#SBATCH -J saxpy           # job name
#SBATCH -o logs/saxpy.o%j       # output and error file name (%j expands to jobID)
#SBATCH -e logs/saxpy.e%j       # output and error file name (%j expands to jobID)
#SBATCH -N 1               # total number of node
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gtx             # queue (partition) -- normal, development, etc.
#SBATCH -t 00:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=jayeshjo1@utexas.edu
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
module load gcc
module load cuda
./cudaSaxpy -n 500000000 -p 1 -i 4
./cudaSaxpyUvm -n 500000000 -p 1 -i 4
./cudaSaxpyStream -n 500000000 -p 1 -i 4
