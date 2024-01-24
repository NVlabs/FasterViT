#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x

set -o pipefail

OUTPUT_DIR=$1

# clean up *.pyc files
rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}

# tar src to avoid future editing
cleanup() {
  echo "Packing source code"
  rmpyc
  # tar -zcf models datasets util main.py engine.py eval.py submit.py --remove-files
  echo " ...Done"
}


pushd $OUTPUT_DIR
trap cleanup EXIT

args=$(cat *.args)
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py ${args} --resume checkpoint.pth --output_dir . |& tee -a resume.log
popd
