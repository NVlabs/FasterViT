# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


import argparse
from collections import defaultdict
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=Path)
parser.add_argument('output_dir', type=Path)
parser.add_argument('--t_min', default=20)
parser.add_argument('--t_max', default=100)
args = parser.parse_args()


class FindUnionSet(dict):
    def find(self, src):
        if src in self:
            return self.find(self[src])
        return src

    def merge(self, dst, src):
        self[self.find(src)] = self.find(dst)


for seq in os.listdir(args.input_dir):
    print(args.input_dir / seq)
    with open(args.input_dir / seq) as f:
        lines = f.readlines()
    instance_timestamps = defaultdict(list)
    for line in lines:
        f_id, id = map(int, line.split(',')[:2])
        instance_timestamps[id].append(f_id)
    instances = list(instance_timestamps.keys())
    fid_map = FindUnionSet()
    for i in instances:
        for j in instances:
            if fid_map.find(i) == fid_map.find(j):
                continue
            end_t = max(instance_timestamps[i])
            start_t = min(instance_timestamps[j])
            if sum([0 <= start_t - max(pts) < args.t_max for pts in instance_timestamps.values()]) > 1:
                continue
            if sum([0 <= min(pts) - end_t < args.t_max for pts in instance_timestamps.values()]) > 1:
                continue
            dt = start_t - end_t
            if args.t_min < dt < args.t_max:
                print(f"{i}<-{j}", end_t, start_t, start_t - end_t)
                fid_map.merge(i, j)

    os.makedirs(args.output_dir / 'tracker', exist_ok=True)
    with open(args.output_dir / 'tracker' / seq, 'w') as f:
        for line in lines:
            f_id, id, *info = line.split(',')
            id = str(fid_map.find(int(id)))
            f.write(','.join([f_id, id, *info]))
