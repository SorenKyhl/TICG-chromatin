#!/bin/bash

# creates a copy of the input file with each line
# of the input file duplicated $factor times

in=$1
factor=$2

awk '{for(i=0;i<$factor;i++)print}' $in
