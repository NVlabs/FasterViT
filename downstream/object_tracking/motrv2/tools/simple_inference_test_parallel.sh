#!/bin/bash

set -x
set -o pipefail

cd /app/maglev-push-clean-code/motrv2

args=$(cat "$1")
input_directory="$2"
total_gpus="$3"
exp_name="$4"

files=("$input_directory"/checkpoint0*.pth)
num_files=${#files[@]}

if [ -d "$input_directory" ]; then
    for (( i=0; i<$num_files; i+=total_gpus )); do
        gpu_idx=0
        for (( j=$i; j<$num_files && j<$((i+total_gpus)); j++ )); do
            file="${files[$j]}"
            if [ -e "$file" ]; then
                CUDA_VISIBLE_DEVICES="$gpu_idx" python3 submit_dance_test.py ${args} --resume "$file" --exp_name "$exp_name" --mot_path $5 --det_db $6 &

                gpu_idx=$(( (gpu_idx + 1) % total_gpus ))
            fi
        done
        wait  
    done
    wait 
else
    echo "Error: '$input_directory' is not a directory."
fi