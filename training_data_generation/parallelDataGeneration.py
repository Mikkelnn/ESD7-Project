import multiprocessing as mp
import os
import uuid
import time
import shutil
import numpy as np
# import tensorflow as tf
import random as rn

# Import your simulation function from separate file
from simulation import runSimulation
from dsp_mini import range_doppler_fft

# -------------------------------
# Disk Monitor
# -------------------------------
def disk_monitor(root_path, flag, check_interval=1):
    """Continuously update flag: True if disk < 90%, False if >= 90%"""
    _, init_disk_used, _ = shutil.disk_usage(root_path)
    while flag.value:
        total, used, free = shutil.disk_usage(root_path)
        # used_percent = used / total * 100
        # flag.value = used_percent < 98

        disk_used_Byte = (used - init_disk_used)
        flag.value = disk_used_Byte < 100e9 # 100GB in bytes 

        # path =  os.path.join(root_path, "input")
        # flag.value = sum(1 for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) < 25
        time.sleep(check_interval)
    print("Disk monitor exiting.")

# -------------------------------
# Simulation Utilities
# -------------------------------
def generate_batch(worker_seed):
    """Generate a random batch of targets"""
    rn.seed(worker_seed + int(time.time() * 1000) % 1000000)
    n_targets = rn.randint(0, 1)
    data = np.zeros((n_targets, 3)) #, dtype=np.int16)
    for j in range(n_targets):
        data[j, 0] = rn.randint(100, 800)   # range
        data[j, 1] = rn.randint(0, 7500)    # velocity
        data[j, 2] = 0 #rn.randint(-15, 15)    # angle
    return data

# def serialize_example_binary(data: np.ndarray):
#     """Serialize a numpy array to raw-binary TFRecord Example"""
#     feature = {
#         "raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()]))
#     }
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()

# -------------------------------
# Worker
# -------------------------------
def run_worker(root_path, worker_id, disk_flag):
    # os.makedirs(root_path, exist_ok=True)
    params_path = os.path.join(root_path, "labels")
    baseband_path = os.path.join(root_path, "input")
    print(f"Worker {worker_id} started (PID {os.getpid()})")

    while disk_flag.value:  # read shared flag instead of calling disk_usage
        local_uuid = uuid.uuid4()
        # Generate random parameters
        params = generate_batch(worker_seed=worker_id)

        # Call your simulation function
        baseband = runSimulation(params)
        baseband = range_doppler_fft(baseband)
        baseband = baseband / np.max(baseband)

        # Write params as raw-binary TFRecord file with UUID
        # Normalize params
        if len(params) == 0:
             params = [0, 0] # output for no targets
        else:
             params = [params[0][0] / 800.0, params[0][1] / 7500.0] # normalize for a target

        # params_filename = os.path.join(params_path, f"{local_uuid}.tfrecord")
        # with tf.io.TFRecordWriter(params_filename) as writer:
        #     writer.write(serialize_example_binary(params))
        params_filename = os.path.join(params_path, f"{local_uuid}.npy")
        with open(params_filename, 'wb') as f:
                np.save(f, params)

        # Write result as raw-binary TFRecord file with UUID
        # baseband_filename = os.path.join(baseband_path, f"{local_uuid}.tfrecord")
        # with tf.io.TFRecordWriter(baseband_filename) as writer:
        #     writer.write(serialize_example_binary(baseband))
        baseband_filename = os.path.join(baseband_path, f"{local_uuid}.npy")
        with open(baseband_filename, 'wb') as f:
                np.save(f, baseband)

    print(f"Worker {worker_id} stopped (disk >= 90%)")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    root_path = "../../training_data"  # Change to desired folder
    os.makedirs(os.path.join(root_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "input"), exist_ok=True)

    # Shared flag to indicate if disk has space
    with mp.Manager() as manager:
        disk_flag = manager.Value('b', True) # True = disk < 90%

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
