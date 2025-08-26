import tensorflow as tf
import numpy as np
import os

# Schema for parsing our serialized examples
_feature_description = {
    "raw": tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(example_proto):
    """Parse a single TFRecord example."""
    return tf.io.parse_single_example(example_proto, _feature_description)

def load_tfrecord_to_numpy(filename, dtype=np.int16, cols=3):
    """
    Reads a single TFRecord file (1 Example) and returns a numpy array.
    
    Args:
        filename (str): Path to TFRecord file.
        dtype (np.dtype): Data type used when saving (default: np.int16).
        cols (int): Number of columns (default: 3 for [range, velocity, angle]).
    
    Returns:
        np.ndarray: Decoded simulation result.
    """
    raw_dataset = tf.data.TFRecordDataset([filename])
    for raw_record in raw_dataset:
        example = tf.io.parse_single_example(raw_record, _feature_description)
        raw_bytes = example["raw"].numpy()
        baseband = np.frombuffer(raw_bytes, dtype=dtype)
        return baseband.reshape(-1, cols)

def load_all_tfrecords(folder, dtype=np.int16, cols=3):
    """
    Reads all TFRecord files in a folder and returns a list of numpy arrays.
    
    Args:
        folder (str): Folder path containing TFRecord files.
        dtype (np.dtype): Data type used when saving (default: np.int16).
        cols (int): Number of columns (default: 3).
    
    Returns:
        List[np.ndarray]: List of simulation results.
    """
    arrays = []
    for fname in os.listdir(folder):
        if fname.endswith(".tfrecord"):
            baseband = load_tfrecord_to_numpy(os.path.join(folder, fname), dtype, cols)
            arrays.append(baseband)
    return arrays
 