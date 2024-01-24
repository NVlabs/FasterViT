# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

python tools/merge_dance_tracklets.py $1 $2

# python3 ../TrackEval/scripts/run_mot_challenge.py \
#     --SPLIT_TO_EVAL val  \
#     --METRICS HOTA \
#     --GT_FOLDER /data/datasets/DanceTrack/val \
#     --SEQMAP_FILE seqmap \
#     --SKIP_SPLIT_FOL True \
#     --TRACKER_SUB_FOLDER tracker \
#     --TRACKERS_TO_EVAL $2 \
#     --USE_PARALLEL True \
#     --NUM_PARALLEL_CORES 8 \
#     --PLOT_CURVES False \
#     --TRACKERS_FOLDER '' | tee -a $2/eval.log
