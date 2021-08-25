#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
chi="-1&2\\2&-1"
k=2
dataFolder="/project2/depablo/erschultz/dataset_08_24_21"

cd ~/TICG-chromatin
# sbatch bin/random1.sh 1 50 $k $chi $dataFolder
# sbatch bin/random2.sh 51 100 $k $chi $dataFolder
# sbatch bin/random3.sh 101 150 $k $chi $dataFolder
# sbatch bin/random4.sh 151 200 $k $chi $dataFolder
# sbatch bin/random5.sh 201 250 $k $chi $dataFolder
# sbatch bin/random6.sh 251 300 $k $chi $dataFolder
# sbatch bin/random7.sh 301 350 $k $chi $dataFolder
# sbatch bin/random8.sh 351 400 $k $chi $dataFolder
# sbatch bin/random9.sh 401 450 $k $chi $dataFolder
sbatch bin/random10.sh 451 500 $k $chi $dataFolder
sbatch bin/random11.sh 501 550 $k $chi $dataFolder
sbatch bin/random12.sh 551 600 $k $chi $dataFolder
sbatch bin/random13.sh 601 650 $k $chi $dataFolder
sbatch bin/random14.sh 651 700 $k $chi $dataFolder
sbatch bin/random15.sh 701 750 $k $chi $dataFolder
sbatch bin/random16.sh 751 800 $k $chi $dataFolder
