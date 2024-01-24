# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


from glob import glob
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from tqdm import tqdm

det_db = {}
to_cache = []

for file in glob("/data/Dataset/mot/crowdhuman/train_image/*.txt"):
    to_cache.append(file)

for file in glob("/data/Dataset/mot/DanceTrack/*/*/img1/*.txt"):
    to_cache.append(file)

for file in glob("/data/Dataset/mot/MOT17/images/*/*/img1/*.txt"):
    to_cache.append(file)

for file in glob("/data/Dataset/mot/MOT20/train/*/img1/*.txt"):
    to_cache.append(file)

for file in glob("/data/Dataset/mot/HIE20/train/*/img1/*.txt"):
    to_cache.append(file)

pbar = tqdm(total=len(to_cache))

mutex = Lock()
def cache(file):
    with open(file) as f:
        tmp = [l for l in f]
    with mutex:
        det_db[file] = tmp
        pbar.update()

with ThreadPoolExecutor(max_workers=48) as exe:
    for file in to_cache:
        exe.submit(cache, file)

with open("/data/Dataset/mot/det_db_oc_sort_full.json", 'w') as f:
    json.dump(det_db, f)

