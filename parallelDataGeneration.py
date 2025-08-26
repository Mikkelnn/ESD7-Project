import multiprocessing as mp
import os
import uuid
import time
import shutil
import numpy as np
import tensorflow as tf
import random as rn

# Import your simulation function from separate file
from parallelDataGeneration import simulate

# -------------------------------
# Disk Monitor
# -------------------------------
def disk_monitor(root_path, flag, check_interval=1):
    """Continuously update flag: True if disk < 90%, False if >= 90%"""
    while flag.value:
        total, used, free = shutil.disk_usage(root_path)
        used_percent = used / total * 100
        flag.value = used_percent < 90
        time.sleep(check_interval)
    print("Disk monitor exiting.")

# -------------------------------
# Simulation Utilities
# -------------------------------
def generate_batch(worker_seed):
    """Generate a random batch of targets"""
    rn.seed(worker_seed + int(time.time() * 1000) % 1000000)
    n_targets = rn.randint(0, 3)
    data = np.zeros((n_targets, 3), dtype=np.int16)
    for j in range(n_targets):
        data[j, 0] = rn.randint(100, 800)    # range
        data[j, 1] = rn.randint(0, 7500)    # velocity
        data[j, 2] = rn.randint(-15, 15)    # angle
    return data

def serialize_example_binary(data: np.ndarray):
    """Serialize a numpy array to raw-binary TFRecord Example"""
    feature = {
        "raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# -------------------------------
# Worker
# -------------------------------
def run_worker(root_path, worker_id, disk_flag):
    os.makedirs(root_path, exist_ok=True)
    print(f"Worker {worker_id} started (PID {os.getpid()})")

    while disk_flag.value:  # read shared flag instead of calling disk_usage
        # Generate random parameters
        params = generate_batch(worker_seed=worker_id)

        # Call your simulation function
        result = simulate(params)

        # Write result as raw-binary TFRecord file with UUID
        filename = os.path.join(root_path, f"{uuid.uuid4()}.tfrecord")
        with tf.io.TFRecordWriter(filename) as writer:
            writer.write(serialize_example_binary(result))

    print(f"Worker {worker_id} stopped (disk >= 90%)")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    root_path = "./sim_output"  # Change to desired folder
    os.makedirs(root_path, exist_ok=True)

    # Shared flag to indicate if disk has space
    disk_flag = mp.Value('b', True)  # True = disk < 90%

    # Start disk monitor process
    monitor = mp.Process(target=disk_monitor, args=(root_path, disk_flag))
    monitor.start()

    # Start worker pool
    n_workers = mp.cpu_count()
    print(f"Starting {n_workers} workers...")
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(run_worker, [(root_path, i, disk_flag) for i in range(n_workers)])

    # Cleanup
    monitor.join()
    print("All workers stopped, simulation complete.")
