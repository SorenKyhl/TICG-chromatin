#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_01_17_22/samples


rm -r sample1 &
rm -r sample1* &
rm -r sample2 &
rm -r sample3 &
rm -r sample4 &
rm -r sample5 &

wait

cd /project2/depablo/erschultz
rm -r dataset_01_17_22
