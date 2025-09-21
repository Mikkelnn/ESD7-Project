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
log.info("Finalised importing AI libs")

#TODO run on a Linux machine or WSL, due to tensorflow lacking gpu support on windows
#tf.config.optimizer.set_jit(True)  #Global jit optimiser

class AiHandler():
    """Class for helping get Tensorflow hardware and helper functions"""
    def __init__(self):
        """Initiates AI Handler"""
        self.time_of_import = time.localtime()
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

def _main():
    ai_handler = AiHandler()
    if ai_handler.tf.test.is_built_with_cuda():
        ai_handler.log.info("TensorFlow is built with CUDA support.")
    else:
        ai_handler.log.info("TensorFlow is NOT built with CUDA support.")

if __name__ == "__main__":
    _main()