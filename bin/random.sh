#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
chi="-1&1\\1&0"
# chi='none'
k=2
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_10_27_21"
startSample=1
relabel='none'
tasks=20
samples=400
samplesPerTask=$(($samples / $tasks))
samplesPerTask=1
diag='true'
local=0

if [ $local -eq 1 ]
then
  dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  tasks=1
  samplesPerTask=1
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  scratchDir="/scratch/midway2/erschultz"
  source activate python3.8_pytorch1.8.1_cuda10.2
fi

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

cd ~/TICG-chromatin

for i in 0 1 2 3 4
do
  startSampleI=$(( $startSample + $samples * $i ))
  # echo $startSampleI
  bash ~/TICG-chromatin/bin/random${i}.sh $chi $k $m $dataFolder $startSampleI $relabel $tasks $samples $samplesPerTask $diag $scratchDir $i
done
