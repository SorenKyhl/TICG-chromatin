#!/bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

# this example shows a maximum entropy procedure which
# matches the parameters of a "ground truth" simulation (iteration 0),
# the ground truth chi parameters are specified by the first line of
# chis.txt and chis_diag.txt:
#
# chis.txt (line 1):  -1 1 -1
# chis_diag.txt (line 1):  np.linspace(0,0.1,16)
#
# the maximum entropy procedure begins in iteration1,
# with initial chi paramters specified by the second line of
# chis.txt and chis_diag.txt
#
# chis.txt (line 2): 0 0 0
# chis_diag.txt (line 2): np.zeros(16)

# if on midway
# module load jq
# module load python

# using all default parameters:

# activate python
source activate python3.8_pytorch1.8.1_cuda10.2

cd ~/TICG-chromatin/maxent
bin/run.sh
