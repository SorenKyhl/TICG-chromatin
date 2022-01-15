#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd /project2/depablo/erschultz/dataset_01_15_22


rm -r sample202* &
rm -r sample201* &
rm -r sample2001 &
rm -r sample2002 &
rm -r sample2003 &
rm -r sample2004 &
rm -r sample2005 &
rm -r sample2006 &
rm -r sample2007 &
rm -r sample2008 &
rm -r sample2009 &

wait
