#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat $1)
output_dir="$2"
dataset_dir="$3" # /develop/dl_tracking/Dataset
db_path="$4" # /develop/dl_tracking/Dataset/det_db_download/det_db_motrv2.json
gpu_num="$5" # 8

python -m torch.distributed.launch --nproc_per_node=$gpu_num --use_env main.py ${args} --output_dir $output_dir --mot_path $dataset_dir --det_db $db_path |& tee -a $output_dir/output.log