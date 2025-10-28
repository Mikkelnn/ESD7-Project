## Run this script from root context ie the level where "training_data" and "validate_data" exist

import os, random, shutil

src_root = "training_data"
dst_root = "validate_data"
subdirs = ["input", "labels"]
ratio = 0.15 # validation ratio

for sub in subdirs:
    src = os.path.join(src_root, sub)
    dst = os.path.join(dst_root, sub)
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    n = int(len(files) * ratio)
    for f in random.sample(files, n):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
