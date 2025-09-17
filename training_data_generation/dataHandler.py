import polars as ps
import numpy as np
import string
import os
import glob

class DataHandler():

    def __init__(self):
        pass

    def print_numpy(self, numpy_array_path):

        with open(numpy_array_path, 'rb') as f:
            arr = np.load(f)
        
        print(arr)

    def unpack_params(self, array) -> tuple[dict[str, ps.Float32], dict[str, ps.Float32]]:
        # Ensure output values are Python floats for compatibility with polars DataFrame
        target1 = {
            "Range": float(array[0, 0]),
            "Velocity": float(array[0, 1]),
            "Angle": float(array[0, 2])
        }
        if array.shape[0] > 1:
            target2 = {
                "Range": float(array[1, 0]),
                "Velocity": float(array[1, 1]),
                "Angle": float(array[1, 2])
            }
        else:
            target2 = {"Range": None, "Velocity": None, "Angle": None}
        return target1, target2

    def unpack_baseband(self, array) -> dict[str, ps.Array]:
        # Assumes array shape is (6, N, M) where 6 is number of antennas
        antennas = {}
        for i in range(array.shape[0]):
            antennas[f"Antenna{i+1}"] = array[i]
        return antennas

    def numpy_to_parquet(self, baseband_array_path, params_array_path):

        with open(baseband_array_path, 'rb') as f_baseband:
            baseband_array = np.load(f_baseband)
        with open(params_array_path, 'rb') as f_params:
            params_array = np.load(f_params)

        target1, target2 = self.unpack_params(params_array)
        radar = self.unpack_baseband(baseband_array)

        parameters = ps.DataFrame({
            "target1_Range": [target1.get("Range")],
            "target1_Velocity": [target1.get("Velocity")],
            "target1_Angle": [target1.get("Angle")],
            "target2_Range": [target2.get("Range")],
            "target2_Velocity": [target2.get("Velocity")],
            "target2_Angle": [target2.get("Angle")],
        }, schema={
            "target1_Range": ps.Float32,
            "target1_Velocity": ps.Float32,
            "target1_Angle": ps.Float32,
            "target2_Range": ps.Float32,
            "target2_Velocity": ps.Float32,
            "target2_Angle": ps.Float32,
        }, strict=True).rechunk()

        # Split each antenna's complex array into real and imaginary columns
        baseband_data = {}
        for i in range(baseband_array.shape[0]):
            baseband_data[f"Antenna{i+1}_real"] = radar[f"Antenna{i+1}"].real.flatten().astype(np.float32)
            baseband_data[f"Antenna{i+1}_imag"] = radar[f"Antenna{i+1}"].imag.flatten().astype(np.float32)
        baseband = ps.DataFrame(
            baseband_data,
            schema={k: ps.Float32 for k in baseband_data.keys()},
            strict=True
        ).rechunk()



        # Save DataFrames to parquet files in respective subfolders
        output_dir = os.path.join(os.getcwd(), "parquet")
        os.makedirs(os.path.join(output_dir, "baseband"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "params"), exist_ok=True)

        # Generate file names based on input paths
        baseband_filename = os.path.splitext(os.path.basename(baseband_array_path))[0] + ".parquet"
        params_filename = os.path.splitext(os.path.basename(params_array_path))[0] + ".parquet"

        baseband_path = os.path.join(output_dir, "baseband", baseband_filename)
        params_path = os.path.join(output_dir, "params", params_filename)

        baseband.write_parquet(
            baseband_path,
            compression="zstd",
            compression_level=3,
            row_group_size=128*1024*1024
        )
        parameters.write_parquet(
            params_path,
            compression="zstd",
            compression_level=3,
            row_group_size=128*1024*1024
        )

    def print_parquet(self, parquet_file_path):
        if not os.path.exists(parquet_file_path):
            print(f"File not found: {parquet_file_path}")
            return
        df = ps.read_parquet(parquet_file_path)
        print(df)

def _main():
    handler = DataHandler()
    baseband_files = sorted(glob.glob("../training_data_generation/sim_output/baseband/*.npy"))
    params_files = sorted(glob.glob("../training_data_generation/sim_output/params/*.npy"))

    error_count = 0
    # Ensure matching files by name
    for idx, (baseband_path, params_path) in enumerate(zip(baseband_files, params_files)):
        try:
            handler.numpy_to_parquet(baseband_path, params_path)
            # Print every 50th processed parquet file
            if (idx + 1) % 50 == 0:
                baseband_name = os.path.splitext(os.path.basename(baseband_path))[0]
                params_name = os.path.splitext(os.path.basename(params_path))[0]
                handler.print_parquet(f"./parquet/params/{params_name}.parquet")
                handler.print_parquet(f"./parquet/baseband/{baseband_name}.parquet")
                print(f"IndexError occurred {error_count} times out of {len(params_files)}.")
        except IndexError:
            error_count += 1

    # Print the last processed parquet files
    if baseband_files and params_files:
        last_baseband = os.path.splitext(os.path.basename(baseband_files[-1]))[0]
        last_params = os.path.splitext(os.path.basename(params_files[-1]))[0]
        handler.print_parquet(f"./parquet/params/{last_params}.parquet")
        handler.print_parquet(f"./parquet/baseband/{last_baseband}.parquet")

if __name__ == "__main__":
    _main()