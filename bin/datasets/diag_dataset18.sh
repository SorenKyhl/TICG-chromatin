#! /bin/bash
#SBATCH --job-name=fds18
#SBATCH --output=logFiles/fixed_diag_dataset18.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000

source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9
sourceFile=$1
source $sourceFile

python3 ~/TICG-chromatin/bin/datasets/run_new.py --start $2 --end $3 --jobs $4 --odir_start $5 --data_folder $dataFolder --scratch $scratchDir --m $m --n_sweeps $nSweeps --dump_frequency $dumpFrequency --TICG_seed $TICGSeed --phi_chromatin $phiChromatin --volume $volume --bead_vol $beadVol --bond_length $bondLength --track_contactmap $trackContactMap
