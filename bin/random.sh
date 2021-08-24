#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
chi="-1&1\\2&-1"
k=2
dataFolder="/project2/depablo/erschultz/dataset_08_24_21"

cd ~/TICG-chromatin
sbatch bin/random1.sh 1201 1250 $k $chi $dataFolder
sbatch bin/random2.sh 1251 1300 $k $chi $dataFolder
sbatch bin/random3.sh 1301 1350 $k $chi $dataFolder
sbatch bin/random4.sh 1351 1400 $k $chi $dataFolder
sbatch bin/random5.sh 1401 1450 $k $chi $dataFolder
sbatch bin/random6.sh 1451 1500 $k $chi $dataFolder
sbatch bin/random7.sh 1501 1550 $k $chi $dataFolder
sbatch bin/random8.sh 1551 1600 $k $chi $dataFolder
sbatch bin/random9.sh 1601 1650 $k $chi $dataFolder
sbatch bin/random10.sh 1651 1700 $k $chi $dataFolder
sbatch bin/random11.sh 1701 1750 $k $chi $dataFolder
sbatch bin/random12.sh 1751 1800 $k $chi $dataFolder
sbatch bin/random13.sh 1801 1850 $k $chi $dataFolder
sbatch bin/random14.sh 1851 1900 $k $chi $dataFolder
sbatch bin/random15.sh 1901 1950 $k $chi $dataFolder
sbatch bin/random16.sh 1951 2000 $k $chi $dataFolder
