#! /bin/bash

# necessary to ensure log files are in right place
cd ~/TICG-chromatin

sbatch bin/latex.sh
sbatch bin/latex2.sh
sbatch bin/latex3.sh
sbatch bin/latex4.sh
