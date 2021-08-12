#!/bin/bash

# updates the config.json file with new chis from chi.txt file
# usage : update_chi iteration

iteration=$1
proj_bin=$2

line_num=$(($iteration+1))
target="iteration$1/config.json"

chis_data_string="$(sed -n ${line_num}p chis.txt)"
IFS=' '
read -ra chis <<< $chis_data_string

letters=(A B C D E F G H I J K L M)
ntypes=$(jq .nspecies $target)
counter=0

for ((i=0 ; i<$ntypes; i++))
do
    for ((j=$i ; j<$ntypes; j++))
    do
        chistring="chi${letters[$i]}${letters[$j]}"

	    python3 $proj_bin/jsed.py $target $chistring ${chis[$counter]} f

        counter=$((counter+1))
    done
done

