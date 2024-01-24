#!/bin/bash

set -x
set -o pipefail

args=$(cat "$1")
input_directory="$2"
total_gpus="$3"
exp_name="$4"

files=("$input_directory"/*.pth)
num_files=${#files[@]}

if [ -d "$input_directory" ]; then
    for (( i=0; i<$num_files; i+=total_gpus )); do
        gpu_idx=0
        for (( j=$i; j<$num_files && j<$((i+total_gpus)); j++ )); do
            file="${files[$j]}"
            if [ -e "$file" ]; then
                CUDA_VISIBLE_DEVICES="$gpu_idx" python3 submit_dance.py ${args} --resume "$file" --exp_name "$exp_name" &
                gpu_idx=$(( (gpu_idx + 1) % total_gpus ))
            fi
        done
        wait
    done
    wait
else
    echo "Error: '$input_directory' is not a directory."
fi