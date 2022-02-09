#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
# chi="-1&1\\1&0"
# chi="-0.5000 & 0.9000 & -0.8000 & -0.6000 & 0.2000 & 0.8000 & -0.4000 & 0.9000 & 0.3000 & -0.70000 \\
# 0.0000 & 0.2000 & 0.3000 & 0.2000 & 0.1000 & 0.7000 & 0.0000 & 0.3000 & 0.6000 & 0.20000 \\
# 0.0000 & 0.0000 & -0.7000 & -0.4000 & -0.3000 & 0.3000 & 0.1000 & 0.0000 & 0.0000 & 0.40000 \\
# 0.0000 & 0.0000 & 0.0000 & -0.3000 & 0.9000 & 0.4000 & 0.0000 & -0.2000 & -0.4000 & -0.50000 \\
# 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.7000 & 0.9000 & 0.7000 & 0.1000 & -0.6000 & -0.30000 \\
# 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.1000 & 1.0000 & 0.4000 & 0.5000 & -0.10000 \\
# 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.8000 & 1.0000 & 0.5000 & 0.00000 \\
# 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & -0.0000 & 0.8000 & -0.80000 \\
# 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & -0.7000 & 0.60000 \\
# 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.50000"
chi='polynomial'
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_01_17_22"
startSample=1
relabel='none'
startNode=0
nodes=20
tasks=20
samples=4000
diag='true'
nSweeps=1000000
pSwitch=0.05
minChi=-1
maxChi=1
fillDiag='none'
chiSeed='none'
maxDiagChi=0.2
local='true'

if [ $local = 'true' ]
then
  dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  startSample=1
  nSweeps=2000
  nodes=1
  tasks=3
  samples=3
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  scratchDir="/scratch/midway2/erschultz"
  source activate python3.8_pytorch1.8.1_cuda10.2
fi

samplesPerNode=$(( $samples / $nodes ))
samplesPerTask=$(( $samplesPerNode / $tasks ))
echo "samples per node" $samplesPerNode
echo "samples per task" $samplesPerTask

cd ~/TICG-chromatin/src
make
mv TICG-engine ..

cd ~/TICG-chromatin

for i in $( seq $startNode $(( $nodes - 1 + $startNode )) )
do
  startSampleI=$(( $startSample + $samplesPerNode * $i ))
  endSampleI=$(( $startSampleI + $samplesPerNode - 1 ))
  echo "TICG${i}" $startSampleI $endSampleI
  bash ~/TICG-chromatin/bin/random/random${i}.sh $chi $k $m $dataFolder $startSampleI $relabel $tasks $samplesPerNode $samplesPerTask $diag $scratchDir $i $nSweeps $pSwitch $minChi $maxChi $fillDiag $chiSeed $maxDiagChi
done
