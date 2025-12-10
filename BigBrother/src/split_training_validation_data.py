## Run this script from root context ie the level where "training_data" and "validate_data" exist

import os, random, shutil

src_root = "one/training_data"
dst_root = "one/validate_data"
ratio = 0.15 # validation ratio

src_inputs = os.path.join(src_root, "input")
src_labels = os.path.join(src_root, "labels")
dst_inputs = os.path.join(dst_root, "input")
dst_labels = os.path.join(dst_root, "labels")

os.makedirs(dst_inputs, exist_ok=True)
os.makedirs(dst_labels, exist_ok=True)

files = os.listdir(src_inputs) # onlly filenames
n = int(len(files) * ratio)
for f in random.sample(files, n):
    shutil.move(os.path.join(src_inputs, f), os.path.join(dst_inputs, f))
    shutil.move(os.path.join(src_labels, f), os.path.join(dst_labels, f))
