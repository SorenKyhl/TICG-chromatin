#! /bin/bash

source ~/TICG-chromatin/bin/datasets/dataset_fns.sh

dataset="dataset_03_12_24_imr90"
dataFolder="/project2/depablo/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
#
# dataFolder="/home/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'
# nSweeps=30000
# dumpFrequency=5000

m=512
useL='true'
useS='true'
useD='true'

nSweeps=300000
dumpFrequency=50000
TICGSeed=10
dense='false'
diagBins=512
volume=8
phiChromatin='none'
beadVol=130000
bondLength=200
maxDiagChi='none'
trackContactMap='false'
updateContactDistance='false'
