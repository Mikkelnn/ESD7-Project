from logger import get_logger
log = get_logger()
log.info("Started importing AI libs")
import os #noqa
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   #Floating point unstability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Recompilation of tensorflow ignore
import tensorflow #noqa
import polars #noqa
import numpy #noqa
import time #noqa
import tensorflow.keras as keras #noqa
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
log.info("Finalised importing AI libs")



#TODO run on a Linux machine or WSL, due to tensorflow lacking gpu support on windows
#tf.config.optimizer.set_jit(True)  #Global jit optimiser

class AiHandler():
    """Class for helping get Tensorflow hardware and helper functions"""
    def __init__(self, result_path):
        """Initiates AI Handler"""
        self.time_of_import = time.localtime()        
        self.result_path = result_path / self.set_time_start()
        self.result_path.mkdir(parents=True, exist_ok=True)
        self.tensorboard_logdir = self.result_path / "logs"        
        self.log = get_logger()

        self.tf = tensorflow
        self.np = numpy
        self.ps = polars

        self.tf.debugging.set_log_device_placement(True)

        self.cuda_built = self.tf.test.is_built_with_cuda()
        self.cudnn_loaded = self.tf.test.is_built_with_gpu_support()
        self.cpu_list = self.tf.config.list_physical_devices('CPU')
        self.gpu_list = self.tf.config.list_physical_devices('GPU')
        self.gpu_amount = len(self.gpu_list)
        self.cpu_amount = len(self.cpu_list)

        self.log.info(f"CUDA built: {self.cuda_built}")
        self.log.info(f"cuDNN loaded: {self.cudnn_loaded}")
        self.log.info(f"CPUs {self.cpu_amount}: {self.cpu_list}")
        self.log.info(f"GPUs {self.gpu_amount}: {self.gpu_list}")

    def set_time_start(self):
        """Sets a timestamp for when user marks AI execution to start"""
        self.start_time = time.localtime()
        return time.strftime("%d-%m-%Y_%H:%M:%S", self.start_time)

    def set_time_stop(self):
        """Sets a timestamp for when user marks AI execution to be done"""
        self.stop_time = time.localtime()
        return time.strftime("%d-%m-%Y_%H:%M:%S", self.stop_time)

    def time_diff(self):
        """Returns the difference in seconds between start and stop time."""
        return time.mktime(self.stop_time) - time.mktime(self.start_time)

    def plot_block_diagram(self, model):
        keras.utils.plot_model(
            model,
            to_file=(self.result_path / "model_block_diagram.png"), # Save the plot to a file
            show_shapes=True,             # Display shape information
            show_layer_names=True,        # Display layer names
            show_layer_activations=True,  # Display activation functions
            rankdir="LR"                  # Orientation: TB=Top-to-Bottom, LR=Left-to-Right
        )

    def compile_model(self, model, optimizer="adam", loss="mse", metrics=None):
        """Compile model with defaults or user settings"""
        if metrics is None:
            metrics = ["mae"]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def fit_model(self, model, train_data, val_data=None,
                  epochs=10, batch_size=32, use_tensorboard=True):
        """
        Train model with optional validation data and TensorBoard logging.
        Returns tf.keras.callbacks.History
        """
        callbacks = []

        if use_tensorboard:            
            tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(
                log_dir=self.tensorboard_logdir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard_cb)

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history

    def launch_tensorboard_threaded(self, logdir=None, port=6006):
        """
        Launch TensorBoard in a separate thread (non-blocking).
        Use this before or during training.
        """

        if logdir is None:
            logdir = self.tensorboard_logdir

        self.tb_process = subprocess.Popen([
            "tensorboard", "--logdir", str(logdir), "--host", "0.0.0.0", "--port", str(port)
        ])
        print(f"TensorBoard started at http://localhost:{port}")

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self.__handle_sigint)

    def __handle_sigint(self, sig, frame):
        """Handler for Ctrl+C to terminate TensorBoard and exit cleanly"""
        print("\nCtrl+C detected. Shutting down TensorBoard...")
        if hasattr(self, 'tb_process') and self.tb_process.poll() is None:
            self.tb_process.terminate()
            self.tb_process.wait()
            print("TensorBoard terminated.")
        sys.exit(0)

    def save_model(self, model, name="model"):
        """Save model to result path"""
        path = self.result_path / f"{name}.keras"
        model.save(path)
        return path

    def load_model(self, path):
        """Load model from file"""
        return self.tf.keras.models.load_model(path)

    def predict(self, model, data):
        """Run prediction"""
        return model.predict(data)

    def dataset_from_directory(self, directory, 
                               image_size=(224, 224), 
                               batch_size=32,
                               validation_split=0.2, 
                               subset="training",
                               seed=42):
        """Load dataset from labeled subdirectories (classification use-case)"""
        return self.tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )

    def dataset_from_data_and_labels(
        self,
        data_dir,
        label_dir,
        loader_func_data=None,
        loader_func_label=None,
        batch_size=32,
        shuffle=True,
        seed=42
    ):
        """
        Create dataset from two directories: one for data, one for labels.
        Files are matched by sorted order of filenames.
        """

        data_dir, label_dir = Path(data_dir), Path(label_dir)
        data_files  = sorted(list(data_dir.glob("*")))
        label_files = sorted(list(label_dir.glob("*")))
        assert len(data_files) == len(label_files), "Data and label counts differ"

        if loader_func_data is None:
            loader_func_data = lambda f: np.load(f)  # default expects .npy
        if loader_func_label is None:
            loader_func_label = lambda f: np.load(f)  # default expects .npy

        first_data = loader_func_data(data_files[0])
        first_label = loader_func_label(label_files[0])
        data_shape = first_data.shape
        label_shape = first_label.shape

        def gen():
            for d, l in zip(data_files, label_files):
                yield loader_func_data(d), loader_func_label(l)

        dataset = self.tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                self.tf.TensorSpec(shape=data_shape, dtype=self.tf.float32), # data
                self.tf.TensorSpec(shape=label_shape, dtype=self.tf.float32), # label
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(len(data_files), seed=seed)

        dataset = dataset.batch(batch_size).prefetch(self.tf.data.AUTOTUNE)
        return dataset

def _main():
    ai_handler = AiHandler()
    if ai_handler.tf.test.is_built_with_cuda():
        ai_handler.log.info("TensorFlow is built with CUDA support.")
    else:
        ai_handler.log.info("TensorFlow is NOT built with CUDA support.")

if __name__ == "__main__":
    _main()