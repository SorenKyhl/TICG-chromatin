#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=6:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu

dir='/home/erschultz/dataset_02_20_23/samples'
cd $dir

for i in 1 2 3 4 5 324 981 1936 2834 3464
do
  cd "${dir}/sample${i}"
  rm -r GNN-*-E
done
