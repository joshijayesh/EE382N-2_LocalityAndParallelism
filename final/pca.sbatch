#!/bin/bash
#SBATCH -J pca              # job name
#SBATCH -o pca.o%j          # expands to jobID
#SBATCH -N 1                # total number of node
#SBATCH -n 1                # total number of mpi tasks requested
#SBATCH -p gtx              # queue (partition) — normal, development, etc.
#SBATCH -t 00:30:00         # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=jayeshjo@utexas.edu
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes
#SBATCH -A EE382N-20-Parallelis
module load gcc
module load cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/8.0/lib64
./eigenfaces -s media/att_faces/

