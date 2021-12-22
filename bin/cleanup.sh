#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_12_17_21/samples
cd sample40
rm -r ground_truth-E &
rm -r ground_truth-S &
wait

cd ../sample1230
rm -r ground_truth-E &
rm -r ground_truth-S &
wait

cd ../sample1718
rm -r ground_truth-E &
rm -r ground_truth-S &
wait
