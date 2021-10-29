#! /bin/bash

chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
chi="-1&1\\1&0"
# chi='none'
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_10_27_21"
startSample=1
relabel='none'
nodes=8
tasks=20
samples=2000
diag='true'
nSweeps=1000000
local=1

if [ $local -eq 1 ]
then
  dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  startSample=1
  nodes=1
  tasks=15
  samples=1000
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  scratchDir="/scratch/midway2/erschultz"
  source activate python3.8_pytorch1.8.1_cuda10.2
fi

samplesPerNode=$(( $samples / $nodes ))
samplesPerTask=$(( $samplesPerNode / $tasks ))
echo "samples per node" $samplesPerNode
echo "samples per task" $samplesPerTask

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

cd ~/TICG-chromatin

for i in $( seq 0 $(( $nodes - 1 )) )
do
  startSampleI=$(( $startSample + $samplesPerNode * $i ))
  endSampleI=$(( $startSampleI + $samplesPerNode - 1 ))
  echo $startSampleI $endSampleI
  # bash ~/TICG-chromatin/bin/random${i}.sh $chi $k $m $dataFolder $startSampleI $relabel $tasks $samplesPerNode $samplesPerTask $diag $scratchDir $i $nSweeps
done
