#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat $1)
input_directory="$2"
gpu_idx="$3"
exp_name="$4"

if [ -d "$input_directory" ]; then
    for file in "$input_directory"/*.pth; do
        if [ -e "$file" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_idx python3 submit_dance.py ${args} --resume "$file" --exp_name "$exp_name"
        fi
    done
else
    echo "Error: '$input_directory' is not a directory."
fi
