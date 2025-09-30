import polars as ps
import numpy as np
import os
import glob
import time

class DataHandler():

    def unpack_params(self, array) -> list[dict[str, ps.Float32]]:
        targets = []
        for i in range(array.shape[0]):
            targets.append({
                f"Target{i+1}_range": float(array[i, 0]),
                f"Target{i+1}_velocity": float(array[i, 1]),
                f"Target{i+1}_angle": float(array[i, 2])
            })
        return targets

    def unpack_baseband(self, array) -> dict[str, ps.Array]:
        antennas = {}
        for i in range(array.shape[0]):
            antennas[f"Antenna{i+1}"] = array[i]
        return antennas

    def numpy_to_parquet(self, baseband_array_path, params_array_path):

        with open(baseband_array_path, 'rb') as f_baseband:
            baseband_array = np.load(f_baseband)
        with open(params_array_path, 'rb') as f_params:
            params_array = np.load(f_params)

        targets = self.unpack_params(params_array)
        radar = self.unpack_baseband(baseband_array)

        param_dict = {}
        param_schema = {}
        for idx, target in enumerate(targets):
            tname = f"Target{idx+1}"
            param_dict[f"{tname}_range"] = [target[f"{tname}_range"]]
            param_dict[f"{tname}_velocity"] = [target[f"{tname}_velocity"]]
            param_dict[f"{tname}_angle"] = [target[f"{tname}_angle"]]
            param_schema[f"{tname}_range"] = ps.Float32
            param_schema[f"{tname}_velocity"] = ps.Float32
            param_schema[f"{tname}_angle"] = ps.Float32
        parameters = ps.DataFrame(param_dict, schema=param_schema, strict=True).rechunk().lazy()

        # Split each antenna's complex array into real and imaginary columns
        baseband_data = {}
        for i in range(baseband_array.shape[0]):
            baseband_data[f"Antenna{i+1}_real"] = radar[f"Antenna{i+1}"].real.flatten().astype(np.float32)
            baseband_data[f"Antenna{i+1}_imag"] = radar[f"Antenna{i+1}"].imag.flatten().astype(np.float32)
        baseband = ps.DataFrame(
            baseband_data,
            schema={k: ps.Float32 for k in baseband_data.keys()},
            strict=True
        ).rechunk().lazy()

        # Save DataFrames to parquet files in respective subfolders
        output_dir = os.path.join(os.getcwd(), "parquet")
        os.makedirs(os.path.join(output_dir, "baseband"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "params"), exist_ok=True)

        # Generate file names based on input paths
        baseband_filename = os.path.splitext(os.path.basename(baseband_array_path))[0] + ".parquet"
        params_filename = os.path.splitext(os.path.basename(params_array_path))[0] + ".parquet"

        baseband_path = os.path.join(output_dir, "baseband", baseband_filename)
        params_path = os.path.join(output_dir, "params", params_filename)

        baseband.sink_parquet(
            baseband_path,
            compression="zstd",
            compression_level=3,
            row_group_size=128*1024*1024 # 128 MB
        )
        parameters.sink_parquet(
            params_path,
            compression="zstd",
            compression_level=3,
            row_group_size=128*1024*1024 # 128 MB
        )

    def parquet_to_tfrecord_example(self, parquet_params_file_path, parquet_baseband_file_path): #TODO

        # TODO notes:
        # TFRecord shards: 256 MB (tune 100–500 MB).
        # Input pipeline tensorflow: use num_parallel_reads=tf.data.AUTOTUNE, interleave, map(..., num_parallel_calls=AUTO), prefetch(AUTO).

        params_tfrecord = None
        baseband_tfrecord = None

        return params_tfrecord, baseband_tfrecord

    def print_parquet(self, parquet_file_path):
        if not os.path.exists(parquet_file_path):
            print(f"File not found: {parquet_file_path}")
            return
        df = ps.read_parquet(parquet_file_path)
        print(df)

    def print_numpy(self, numpy_array_path):
        with open(numpy_array_path, 'rb') as f:
            arr = np.load(f)
        print(arr)

def _main():
    handler = DataHandler()
    baseband_files = sorted(glob.glob("../training_data_generation/sim_output/baseband/*.npy"))
    params_files = sorted(glob.glob("../training_data_generation/sim_output/params/*.npy"))

    start_time = time.time()

    for baseband_path, params_path in zip(baseband_files, params_files):
        handler.numpy_to_parquet(baseband_path, params_path)

    # Print every 200th params and baseband parquet file for inspection, along with their numpy pairs
    for i in range(0, len(baseband_files), 400):
        print(f"\n--- Params file: {params_files[i]} ---")
        handler.print_numpy(params_files[i])
        handler.print_parquet(
            os.path.join(
                os.getcwd(),
                "parquet",
                "params",
                os.path.splitext(os.path.basename(params_files[i]))[0] + ".parquet"
            )
        )
        print(f"\n--- Baseband file: {baseband_files[i]} ---")
        handler.print_numpy(baseband_files[i])
        handler.print_parquet(
            os.path.join(
                os.getcwd(),
                "parquet",
                "baseband",
                os.path.splitext(os.path.basename(baseband_files[i]))[0] + ".parquet"
            )
        )

    elapsed = time.time() - start_time
    print(f"\nTotal time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    _main()