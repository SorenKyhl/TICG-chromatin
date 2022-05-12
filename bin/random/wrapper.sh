#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
# chi="0&0\\0&0"
chi='polynomial'
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_05_12_22"
startSample=1
relabel='none'
startNode=0
nodes=20
tasks=20
samples=2000
diag='true'
nSweeps=1000000
pSwitch=0.05
minChi=-1
maxChi=1
fillDiag='none'
chiSeed='none'
maxDiagChi=10
local='false'

if [ $local = 'true' ]
then
  dataFolder="/home/erschultz/dataset_test"
  scratchDir='/home/erschultz/scratch'
  startSample=1
  nSweeps=100000
  nodes=1
  tasks=2
  samples=2
  source activate python3.9_pytorch1.9
else
  scratchDir="/scratch/midway2/erschultz"
  source activate python3.9_pytorch1.9_cuda10.2
fi

samplesPerNode=$(( $samples / $nodes ))
samplesPerTask=$(( $samplesPerNode / $tasks ))
echo "samples per node" $samplesPerNode
echo "samples per task" $samplesPerTask

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

cd ~/TICG-chromatin

for i in $( seq $startNode $(( $nodes - 1 + $startNode )) )
do
  startSampleI=$(( $startSample + $samplesPerNode * $i ))
  endSampleI=$(( $startSampleI + $samplesPerNode - 1 ))
  echo "TICG${i}" $startSampleI $endSampleI
  sbatch ~/TICG-chromatin/bin/random/random${i}.sh $chi $k $m $dataFolder $startSampleI $relabel $tasks $samplesPerNode $samplesPerTask $diag $scratchDir $i $nSweeps $pSwitch $minChi $maxChi $fillDiag $chiSeed $maxDiagChi
done
