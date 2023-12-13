#! /bin/bash

source ~/TICG-chromatin/bin/datasets/dataset_fns.sh

dataset="dataset_11_27_23_hmec"
dataFolder="/project2/depablo/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
#
# dataFolder="/home/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'
# nSweeps=3000
# dumpFrequency=500

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
bondLength=180
maxDiagChi='none'
trackContactMap='false'
updateContactDistance='false'
