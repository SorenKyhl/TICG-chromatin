#!/bin/bash

# $1 : source directory
# $2 : target directory
# takes first and last of chis parameters from source and puts them in target directory

head -1 $1/chis.txt > tmp.txt
tail -1 $1/chis.txt >> tmp.txt
mv tmp.txt $2/chis.txt

head -1 $1/chis_diag.txt > tmp.txt
tail -1 $1/chis_diag.txt >> tmp.txt
mv tmp.txt $2/chis_diag.txt
