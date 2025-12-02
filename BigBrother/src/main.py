from ai_handler import AiHandler
from ntfy import NtfyHandler
from logger import get_logger
from pathlib import Path
from model import defineModel_singel_target_estimate, defineModel_single_target_detector
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import math
import keras.losses as kl
import keras.optimizers as ko

# GENEREL_PATH = Path("../../")
GENEREL_PATH = Path("/scratch")  # Use full path for correct mapping on ai-lab container
RESULTS_PATH = GENEREL_PATH / "results"
TRAINING_DATA_PATH = GENEREL_PATH / "training_data" # "big_training_data"
VALIDATE_DATA_PATH = GENEREL_PATH / "validate_data" # "training_data" 

log = get_logger()
ai_handler = AiHandler(RESULTS_PATH)
ntfy = NtfyHandler("ai_template")


def main():
    log.info(f"PYTHON_NUM_THREADS: {mp.cpu_count()}")
    log.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    log.info(f"TF_NUM_INTRAOP_THREADS: {os.environ.get('TF_NUM_INTRAOP_THREADS')}")
    log.info(f"TF_NUM_INTEROP_THREADS: {os.environ.get('TF_NUM_INTEROP_THREADS')}")

    with ai_handler.strategy.scope():
        model = None
        time_started = 0
        batch_size = 32 # Decrease as model get larger to fit in GPU memory
        epochs = 100
        initial_epoch = 0
        train_on_latest_result = False
        
        max_range = 800 # m
        max_velocity = 7500 # m/s - for now only between zero and 7500 m/s
    
        num_range_out = int(max_range / 10) #
        num_velocity_out = int(max_velocity / 50) #
        # output_size = num_range_out + num_velocity_out

        try:
            time_started = ai_handler.set_time_start()

            if train_on_latest_result:
                (found, initial_epoch, model) = ai_handler.find_latest_model()
                epochs += initial_epoch
                if not found:
                    exit()
            else:
                model = defineModel_singel_target_estimate(num_range_out, num_velocity_out) # defineModel_single_target_detector()
            
            model.summary()

            ai_handler.plot_block_diagram(model)

            # loss = kl.CategoricalFocalCrossentropy(
            #     gamma=2.0,
            #     alpha=0.25
            # )

            ce = kl.CategoricalCrossentropy(reduction=None)
            bce = kl.BinaryCrossentropy()

            def masked_range_loss(y_true, y_pred):
                # y_true = [object_flag, one_hot_range]
                object_flag = y_true[:, :1]                         # shape (batch, 1)
                range_label = y_true[:, 1:]                         # shape (batch, range_bins)
                loss = ce(range_label, y_pred)                      # shape (batch,)
                loss = loss * ai_handler.tf.squeeze(object_flag, axis=-1)      # mask
                return ai_handler.tf.reduce_mean(loss)

            def masked_doppler_loss(y_true, y_pred):
                object_flag = y_true[:, :1]
                doppler_label = y_true[:, 1:]
                loss = ce(doppler_label, y_pred)
                loss = loss * ai_handler.tf.squeeze(object_flag, axis=-1)
                return ai_handler.tf.reduce_mean(loss)

            compiled_model = ai_handler.compile_model(model, 
                                optimizer=ko.Adam(1e-4),                                
                                loss={
                                    "target_present": bce,
                                    "range_head":     masked_range_loss,
                                    "doppler_head":   masked_doppler_loss
                                },
                                loss_weights={
                                    "target_present": 1.0,
                                    "range_head":     1.0,
                                    "doppler_head":   1.0
                                },
                                metrics={
                                    "target_present": ["accuracy"],
                                    "range_head":     ["accuracy"],
                                    "doppler_head":   ["accuracy"]
                                })


            def loader_func_label(f): 
                label = np.load(f) # shape (2,) → [range, velocity]
                
                # if (sum(label) == 0):
                #     return np.array([1,0])
                # else:
                #     return np.array([0,1])
                
                target_present = np.array([0], dtype=np.float32)
                range_label = np.zeros(num_range_out, dtype=np.float32)
                doppler_label = np.zeros(num_velocity_out, dtype=np.float32)

                # --- Object presence ---
                if np.sum(label) != 0:
                    target_present[0] = 1 # target present

                    # --- Scale label to relative bin index ---
                    # Example scaling, adjust factors to your bin definitions
                    label_scaled = np.array([
                        label[0] * num_range_out,    # range scale
                        label[1] * num_velocity_out  # doppler scale
                    ])

                    # --- Floor to nearest bin index ---
                    label_idx = np.floor(label_scaled).astype(int)

                    # --- Clip to valid range ---
                    label_idx[0] = np.clip(label_idx[0], 0, num_range_out - 1)
                    label_idx[1] = np.clip(label_idx[1], 0, num_velocity_out - 1)

                    # --- Create one-hot vectors ---
                    range_label[label_idx[0]] = 1.0
                    doppler_label[label_idx[1]] = 1.0

                range_label = np.concatenate([target_present, range_label], axis=-1)
                doppler_label = np.concatenate([target_present, doppler_label], axis=-1)

                # --- Return as dict for 3-head model ---
                return {
                    "target_present": target_present,
                    "range_head": range_label,
                    "doppler_head": doppler_label
                }

            def loader_func_data(f): 
                data = np.load(f)[... , None]
                return np.nan_to_num(data, nan=0.0)

            labeld_data = ai_handler.dataset_from_data_and_labels(
                data_dir=TRAINING_DATA_PATH / "input",
                label_dir=TRAINING_DATA_PATH / "labels",
                batch_size=batch_size,
                shuffle=True,
                loader_func_label=loader_func_label,
                loader_func_data=loader_func_data
            )
            labeld_validation = ai_handler.dataset_from_data_and_labels(
                data_dir=VALIDATE_DATA_PATH / "input",
                label_dir=VALIDATE_DATA_PATH / "labels",
                batch_size=batch_size,
                shuffle=False,
                loader_func_label=loader_func_label,
                loader_func_data=loader_func_data
            )

            # ai_handler.launch_tensorboard_threaded() # Not supported on AI-LAB
            history = ai_handler.fit_model(
                compiled_model,
                train_data=labeld_data,
                val_data=labeld_validation,
                epochs=epochs,
                batch_size=batch_size,
                initialEpoch=initial_epoch
            )

            ai_handler.save_model(compiled_model)

            acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(acc) + 1)

            for i in epochs:
                log.info(
                    f"Epoch {i}: loss {loss[i - 1]}, validation loss {val_loss[i - 1]}, accuracy {acc[i - 1]}, validation accuracy {val_acc[i - 1]}"
                )

            ai_handler.set_time_stop()

            # Plot and save accuracy figure
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, acc, label="Training Accuracy")
            plt.plot(epochs, val_acc, label="Validation Accuracy")
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(ai_handler.result_path / "accuracy.svg", format="svg")
            plt.savefig(ai_handler.result_path / "accuracy.png", format="png")
            plt.close()

            # Plot and save loss figure
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, loss, label="Training Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(ai_handler.result_path / "loss.svg", format="svg")
            plt.savefig(ai_handler.result_path / "loss.png", format="png")
            plt.close()

            # ntfy.post(  # Remember the message is markdown format
            #     title=f"Results of ML {time_started}",
            #     message=(
            #         f"**Start time:** {time_started}\n"
            #         f"**Time spent:** {ai_handler.time_diff()} seconds\n\n"
            #         f"**Results saved to:** `{ai_handler.result_path}`\n\n"
            #     ),
            # )

            # ntfy.post_image(
            #     ai_handler.result_path / "model_block_diagram.png",
            #     title=f"Model block diagram {time_started}",
            #     compress=True,
            # )
            # ntfy.post_image(
            #     ai_handler.result_path / "loss.png", title="Loss diagram", compress=True
            # )
            # ntfy.post_image(
            #     ai_handler.result_path / "accuracy.png",
            #     title="Accuracy diagram",
            #     compress=True,
            # )
        except Exception as e:
            # pass
            ntfy.post(
                title=f"Error during model training {time_started}",
                message=f"An error occurred: {e}",
            )
            # ntfy.post_image(
            #     ai_handler.result_path / "model_block_diagram.png",
            #     title=f"Model block diagram {time_started}",
            #     compress=True,
            # )


def load_predict():
    modelPath = "results/26-09-2025_12:15:53/sum_diff_model.keras"

    model = ai_handler.load_model(modelPath)
    res = ai_handler.predict(model, [0.1, 0.3])
    print(res)


if __name__ == "__main__":
    # load_predict()
    main()
