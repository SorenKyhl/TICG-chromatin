#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir="/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02"

cd $dir
rm -r sc_contacts

for i in 0 1 2 3
do
  rm -r "${dir}/contact_diffusion_eig2/iteration_${i}/sc_contacts" &
  rm -r "${dir}/contact_diffusion_kNN3/iteration_${i}/sc_contacts" &
done

wait

cd ~/scratch-midway2
rm -r dataset_05_12_12
