from ai_handler import AiHandler
from ntfy import NtfyHandler
from logger import get_logger
from pathlib import Path
from model import *
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
import keras.losses as kl
import keras.optimizers as ko
from tqdm import tqdm
import sklearn.metrics as sklearn
import shutil


# GENEREL_PATH = Path("../../")
GENEREL_PATH = Path("/scratch")  # /scratch # Use full path for correct mapping on ai-lab container
RESULTS_PATH = GENEREL_PATH / "results"
TRAINING_DATA_PATH = GENEREL_PATH / "one/training_data" # "big_training_data"
VALIDATE_DATA_PATH = GENEREL_PATH / "one/validate_data" # "training_data"

log = get_logger()
ai_handler = AiHandler(RESULTS_PATH) # , namedResultDir="15-12-2025_20:51:34"
ntfy = NtfyHandler("ai_template")


def main():
    log.info(f"PYTHON_NUM_THREADS: {mp.cpu_count()}")
    log.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    log.info(f"TF_NUM_INTRAOP_THREADS: {os.environ.get('TF_NUM_INTRAOP_THREADS')}")
    log.info(f"TF_NUM_INTEROP_THREADS: {os.environ.get('TF_NUM_INTEROP_THREADS')}")

    with ai_handler.strategy.scope():
        model = None
        time_started = 0
        batch_size = 1 # Decrease as model get larger to fit in GPU memory
        epochs = 20
        initial_epoch = 0
        train_on_latest_result = False
        
        max_range = 1000 # m
        max_velocity = 7500 # m/s - for now only between zero and 7500 m/s
    
        # num_range_out = int(max_range / 10) #
        # num_velocity_out = int(max_velocity / 50) #
        # output_size = num_range_out + num_velocity_out

        try:
            time_started = ai_handler.set_time_start()

            if train_on_latest_result:
                model = define_sweep_single_localization()
                (found, initial_epoch, model) = ai_handler.find_latest_model(model)
                epochs += initial_epoch
                if not found:
                    exit()
            else:
                # model = defineModel_single_target_detector_doubleConv()
                # model = define_sweep_single_localization()
                model = define_sweep_single_localization_lstm()
                # model = defineModel_single_target_detector_sweep()
                # model = define_robust_model_v2(use_heatmap=False)
                # model = defineModel_singel_target_estimate() # model = defineModel_singel_target_estimate_descreete(num_range_out, num_velocity_out) # defineModel_single_target_detector()
                # model = defineModel_smallCNN()
            
            model.summary()

            # exit()

            # ai_handler.plot_block_diagram(model)

            loss = kl.CategoricalFocalCrossentropy(
                gamma=2.0,
                alpha=0.25
            )

            """ 
                        ce = kl.CategoricalCrossentropy(reduction=None)
                        bce = kl.BinaryCrossentropy()

                        def masked_loss(y_true, y_pred):
                            # y_true = [object_flag, one_hot_range]
                            object_flag = y_true[:, 0]                         # shape (batch, 1)
                            range_label = y_true[:, 1:]                         # shape (batch, range_bins)
                            loss = ce(range_label, y_pred)                      # shape (batch,)
                            loss = loss * ai_handler.tf.squeeze(object_flag)
                            # loss = loss * ai_handler.tf.squeeze(object_flag, axis=-1)      # mask
                            return ai_handler.tf.reduce_sum(loss) / (ai_handler.tf.reduce_sum(object_flag) + 1e-6) # ai_handler.tf.reduce_mean(loss)

                        def range_acc_mask(y_true, y_pred):
                            return ai_handler.tf.keras.metrics.categorical_accuracy(y_true[:, 1:], y_pred)

                        def masked_mse(y_true, y_pred):
                            presence = y_true[:, 0]          # first column = target_present
                            coords_true = y_true[:, 1:]      # remaining = coordinates
                            loss = ai_handler.tf.reduce_mean(ai_handler.tf.square(coords_true - y_pred), axis=-1)
                            loss = loss * presence           # zero out absent targets
                            return ai_handler.tf.reduce_sum(loss) / (ai_handler.tf.reduce_sum(presence) + 1e-6)

                        def masked_mae(y_true, y_pred):
                            presence = y_true[:, 0]
                            coords_true = y_true[:, 1:]
                            loss = ai_handler.tf.reduce_mean(ai_handler.tf.abs(coords_true - y_pred), axis=-1)
                            masked_loss = loss * presence
                            return ai_handler.tf.reduce_sum(masked_loss) / (ai_handler.tf.reduce_sum(presence) + 1e-6)

                        # compiled_model = ai_handler.compile_model(model, 
                        #                     optimizer=ko.Adam(learning_rate=1e-4, clipnorm=1.0),
                        #                     loss={
                        #                         "target_present": bce,
                        #                         "coords": masked_mse,
                        #                         # "heatmap": None
                        #                         # "range_head":     masked_loss,
                        #                         # "doppler_head":   masked_loss
                        #                     },
                        #                     loss_weights={
                        #                         "target_present": 10.0,
                        #                         "coords": 1.0,
                        #                         # "heatmap": 0.0
                        #                         # "range_head":     1.0,
                        #                         # "doppler_head":   1.0
                        #                     },
                        #                     metrics={
                        #                         "target_present": ["accuracy"],
                        #                         "coords": [masked_mae],  # regression error in normalized [0,1] units
                        #                         # "heatmap": []       # optional, usually none
                        #                         # "range_head":     [range_acc_mask],
                        #                         # "doppler_head":   [range_acc_mask]
                        #                     })
            """            

            compiled_model = ai_handler.compile_model(model,
                                    optimizer=ko.Adam(1e-4),
                                    loss=kl.Huber(delta=1.0),
                                    metrics=[
                                        ai_handler.tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                                        ai_handler.tf.keras.metrics.MeanSquaredError(name="MSE")
                                    ]
                                    )

            # compiled_model = ai_handler.compile_model(model,
            #                         optimizer=ko.Adam(1e-4),
            #                         loss=loss,
            #                         metrics=["accuracy"]
            #                         )

            def loader_func_label(f): 
                label = np.load(f) # shape (2,) → [range, velocity]
                return np.array([max_range * label[0], max_velocity * label[1]])  # scale to physical units
                # return np.array([1,0]) if (sum(label) == 0) else np.array([0,1])
                

            def loader_func_data(f): 
                # data = (np.load(f)[0])[... , None]
                data = np.load(f)
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

            # data, lbl = next(iter(labeld_data))
            # print(lbl["target_present"][0])
            # print(lbl["range_head"][0].numpy().sum())
            # print(lbl["doppler_head"][0].numpy().sum())
            # exit()

            # ai_handler.launch_tensorboard_threaded() # Not supported on AI-LAB
            history = ai_handler.fit_model(
                compiled_model,
                train_data=labeld_data,
                val_data=labeld_validation,
                epochs=epochs,
                batch_size=batch_size,
                initialEpoch=initial_epoch
            )

            ai_handler.set_time_stop()
            
            ai_handler.save_model(compiled_model)
            
            mae = history.history["MAE"]
            val_mae = history.history["val_MAE"]
            mse = history.history["MSE"]
            val_mse = history.history["val_MSE"]

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(loss) + 1)

            for i in epochs:
                log.info(
                    f"Epoch {i + initial_epoch}: loss {loss[i - 1]}, validation loss {val_loss[i - 1]}, MAE {mae[i - 1]}, validation MAE {val_mae[i - 1]}, MSE {mse[i - 1]}, validation MSE {val_mse[i - 1]}"
                )

            # Plot and save mae figure
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, mae, label="Training MAE")
            plt.plot(epochs, val_mae, label="Validation MAE")
            plt.title("Model MAE")
            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.legend()
            plt.savefig(ai_handler.result_path / "mae.svg", format="svg")
            plt.savefig(ai_handler.result_path / "mae.png", format="png")
            plt.close()

            # Plot and save mse figure
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, mse, label="Training MSE")
            plt.plot(epochs, val_mse, label="Validation MSE")
            plt.title("Model MSE")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.savefig(ai_handler.result_path / "mse.svg", format="svg")
            plt.savefig(ai_handler.result_path / "mse.png", format="png")
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

            # history_dict = history.history
            # epochs = range(1, len(history_dict["loss"]) + 1)

            # # Log all metrics
            # for i in epochs:
            #     log_line = [f"Epoch {i}:"]
            #     for k, v in history_dict.items():
            #         log_line.append(f"{k}={v[i-1]}")
            #     log.info(", ".join(log_line))

            # # Detect head names
            # heads = sorted({
            #     k.rsplit("_", 1)[0]
            #     for k in history_dict.keys()
            #     if ((k.endswith("_accuracy") or k.endswith("_mae") or k.endswith("_loss")) and not k.startswith("val_"))
            # })

            # # ---- ACCURACY PLOT (all heads) ----
            # plt.figure(figsize=(10, 6))

            # for h in heads:
            #     # Try to find the metric, fallback to loss if metric missing
            #     if f"{h}_accuracy" in history_dict:
            #         train = history_dict[f"{h}_accuracy"]
            #         val = history_dict.get(f"val_{h}_accuracy")
            #         label = f"{h} (accuracy)"
            #     elif f"{h}_mae" in history_dict:
            #         train = history_dict[f"{h}_mae"]
            #         val = history_dict.get(f"val_{h}_mae")
            #         label = f"{h} (MAE)"
            #     else:
            #         continue

            #     plt.plot(epochs, train, "o-", label=f"train {label}")
            #     if val is not None:
            #         plt.plot(epochs, val, "x--", label=f"val {label}")

            # plt.xlabel("Epochs")
            # plt.ylabel("Metric")
            # plt.title("Training Metrics per Head")
            # plt.legend()
            # plt.savefig(ai_handler.result_path / "accuracy.svg", format="svg")
            # plt.savefig(ai_handler.result_path / "accuracy.png", format="png")
            # plt.close()


            # # ---- LOSS PLOT (all heads + total) ----
            # plt.figure(figsize=(10, 6))

            # # Global total loss if present
            # if "loss" in history_dict:
            #     plt.plot(epochs, history_dict["loss"], label="total train loss")
            # if "val_loss" in history_dict:
            #     plt.plot(epochs, history_dict["val_loss"], "--", label="total val loss")

            # for h in heads:
            #     train = history_dict.get(f"{h}_loss")
            #     val = history_dict.get(f"val_{h}_loss")

            #     if train is not None:
            #         plt.plot(epochs, train, label=f"{h} train loss")
            #     if val is not None:
            #         plt.plot(epochs, val, "--", label=f"{h} val loss")

            # plt.xlabel("Epoch")
            # plt.ylabel("Loss")
            # plt.title("Loss per Head")
            # plt.legend()
            # plt.savefig(ai_handler.result_path / "loss.svg", format="svg")
            # plt.savefig(ai_handler.result_path / "loss.png", format="png")
            # plt.close()

            # confusion_matrix()

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
            log.error(f"An error occurred during model training: {e}")

            # ntfy.post(
            #     title=f"Error during model training {time_started}",
            #     message=f"An error occurred: {e}",
            # )
            # ntfy.post_image(
            #     ai_handler.result_path / "model_block_diagram.png",
            #     title=f"Model block diagram {time_started}",
            #     compress=True,
            # )
        finally:
            # copy logfiles
            log.info("Copying log files to result directory")
            src_dir = Path(__file__).parent.parent
            for f in ["log.log", "my_job.err", "my_job.out"]:
                src = os.path.join(src_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, ai_handler.result_path)


def load_predict(modelPath = "results/26-09-2025_12:15:53/sum_diff_model.keras"):
    model = ai_handler.load_model(modelPath)
    res = ai_handler.predict(model, [0.1, 0.3])
    print(res)

def confusion_matrix():
    modelPath = ai_handler.result_path

    model = ai_handler.load_model_directory(modelPath)
    
    def loader_func_data(f): 
        data = np.load(f)[... , None]
        return np.nan_to_num(data, nan=0.0)

    def loader_func_label(f):
        arr = np.load(f)
        # class 0: no debris (sum == 0), class 1: debris present (sum != 0)
        return 0 if (np.sum(arr) == 0) else 1
    
        #return np.array([1,0]) if (sum(np.load(f)) == 0) else np.array([0,1])

    data_dir, label_dir = Path(VALIDATE_DATA_PATH / "input"), Path(VALIDATE_DATA_PATH / "labels")
    data_files  = sorted(str(f) for f in Path(data_dir).glob("*"))
    label_files = sorted(str(f) for f in Path(label_dir).glob("*"))
    assert len(data_files) == len(label_files), "Data and label counts differ"

    # N, TP, FP, TN, FN = len(data_files), 0, 0, 0, 0

    y_true = []
    y_pred = []

    for data_file, label_file in tqdm(zip(data_files, label_files), total=len(data_files)):
        pre = ai_handler.predict(model, loader_func_data(data_file))
        pre_idx = int(np.argmax(pre, axis=-1)) 
        act_idx = loader_func_label(label_file) 

        y_true.append(act_idx)
        y_pred.append(pre_idx)

        #if np.array_equal(act, [0,1]) and np.array_equal(pre, [0,1]):
        #    TP += 1
        #elif np.array_equal(act, [0,1]) and np.array_equal(pre, [1,0]):
        #    FN += 1
        #elif np.array_equal(act, [1,0]) and np.array_equal(pre, [0,1]):
        #    FP += 1
        #elif np.array_equal(act, [1,0]) and np.array_equal(pre, [1,0]):
        #    TN += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #TP /= (TP + FN)
    #FN /= (TP + FN)
    #FP /= (FP + TN)
    #TN /= (FP + TN)
    
    cm_counts = sklearn.confusion_matrix(y_true, y_pred, labels=[1, 0])

    #cm = np.array([[TP, FN], [FP, TN]])
    
    TP, FN = cm_counts[0]
    FP, TN = cm_counts[1]
    
    # Optionally normalize per row (true class)
    cm_norm = cm_counts.astype(float)
    cm_norm[0] /= (TP + FN) if (TP + FN) > 0 else 1.0  # positive class row
    cm_norm[1] /= (FP + TN) if (FP + TN) > 0 else 1.0  # negative class row

    # Plot normalized confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_norm, cmap='viridis', vmin=0.0, vmax=1.0)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center', color='black', fontsize=16)

    plt.xticks([0, 1], ['1', '0'])
    plt.yticks([0, 1], ['1', '0'])
    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.gca().spines[:].set_visible(False)

    plt.savefig(ai_handler.result_path / "confusion_matrix.svg", format="svg")
    plt.close()

    return cm_counts, cm_norm

    # Save
    #plt.figure(figsize=(6, 6))
    #plt.imshow(cm, cmap='viridis')

    # Add numbers in the middle of tiles
    #for i in range(cm.shape[0]):
    #    for j in range(cm.shape[1]):
    #        plt.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='black', fontsize=16)

    # Add axis actual
    #plt.xticks([0, 1], ['1', '0'])
    #plt.yticks([0, 1], ['1', '0'])
    #plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    #plt.gca().xaxis.set_label_position('top')
    #plt.xlabel('Predicted')
    #plt.ylabel('Actual')
    #plt.gca().spines[:].set_visible(False)

    #plt.savefig(ai_handler.result_path / "confusion_matrix.svg", format="svg")
    #plt.close()
    
    #return cm

def roc(modelPath: str):

    model = ai_handler.load_model(modelPath)

    # Prepare validation data
    def loader_func_data(f):
        data = np.load(f)[..., None]
        return np.nan_to_num(data, nan=0.0)

    def loader_func_label(f):
        arr = np.load(f)
        # class 0: no debris (sum == 0), class 1: debris present (sum != 0)
        return 0 if (np.sum(arr) == 0) else 1

    data_dir, label_dir = Path(VALIDATE_DATA_PATH / "input"), Path(VALIDATE_DATA_PATH / "labels")
    data_files  = sorted(str(f) for f in Path(data_dir).glob("*"))
    label_files = sorted(str(f) for f in Path(label_dir).glob("*"))
    assert len(data_files) == len(label_files), "Data and label counts differ"

    y_true = []
    y_score = []

    for data_file, label_file in tqdm(zip(data_files, label_files), total=len(data_files)):
        # Predict probability for class 1 (debris present)
        pred = ai_handler.predict(model, loader_func_data(data_file))
        # If output is shape (2,), use softmax or sigmoid output
        if pred.shape[-1] == 2:
            score = float(pred[0][1]) if pred.ndim == 2 else float(pred[1])
        else:
            score = float(pred.ravel()[0])
        y_score.append(score)
        y_true.append(loader_func_label(label_file))

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    log.info("fpr:", fpr)
    log.info("tpr:", tpr)
    log.info("thresholds:", thresholds)
    log.info("roc_auc:", roc_auc)

    # Plot
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(ai_handler.result_path / "roc_curve.svg", format="svg")
    plt.close()
    print(f"ROC AUC: {roc_auc:.3f}")
    return fpr, tpr, thresholds, roc_auc

def get_validation_predictions():
    modelPath = ai_handler.result_path
    model = define_sweep_single_localization()
    model = ai_handler.load_weights_directory(model, modelPath)

    data_dir, label_dir = Path(VALIDATE_DATA_PATH / "input"), Path(VALIDATE_DATA_PATH / "labels")
    data_files  = sorted(str(f) for f in Path(data_dir).glob("*"))
    label_files = sorted(str(f) for f in Path(label_dir).glob("*"))
    assert len(data_files) == len(label_files), "Data and label counts differ"

    log_path = Path(__file__).parent.parent / "metrics.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for data_file, label_file in tqdm(zip(data_files, label_files), total=len(data_files)):
            predicted_raw = ai_handler.predict(model, data = np.load(data_file))
            label_raw = np.load(label_file)

            predicted_raw_scaled = np.multiply(predicted_raw, [1000, 7500])
            label_raw_scaled = np.multiply(label_raw, [1000, 7500])

            f.write(f"predicted_raw: {predicted_raw}; label_raw: {label_raw}; predicted_raw_scaled: {predicted_raw_scaled}; label_raw_scaled: {label_raw_scaled}\n")


if __name__ == "__main__":
    # load_predict()
    main()
    # _ = confusion_matrix()
    # _ = get_validation_predictions()
