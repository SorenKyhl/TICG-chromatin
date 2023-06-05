#! /bin/bash

source ~/TICG-chromatin/bin/datasets/dataset_fns.sh

dataset=dataset_06_05_23
dataFolder="/project2/depablo/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
#
# dataFolder="/home/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'

m=1024
overwrite=1
useL='true'
useS='true'
useD='true'

nSweeps=150000
dumpFrequency=50000
TICGSeed=10
dense='false'
diagBins=1024
phiChromatin=0.03
beadVol=130000
bondLength=140
maxDiagChi='none'
trackContactMap='true'
