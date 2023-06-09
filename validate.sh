#!/bin/bash
DATA_PATH="/home/ali/Desktop/data_local/ImageNet-Validation/val"
BS=128
checkpoint='/home/ali/Desktop/Vision_Transformers/model_weights/new_faster_vit_models/fastervit_4_224_1k.pth.tar'

python validate.py --model faster_vit_4_224 --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS --input-size 3 224 224

