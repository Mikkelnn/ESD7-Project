from ai_handler import AiHandler
from ntfy import NtfyHandler
from logger import get_logger
from pathlib import Path
from model import defineModel
import os
import matplotlib.pyplot as plt

# GENEREL_PATH = Path("../../")
GENEREL_PATH = Path("/scratch")  # Use full path for correct mapping on ai-lab container
RESULTS_PATH = GENEREL_PATH / "results"
TRAINING_DATA_PATH = GENEREL_PATH / "training_data"
VALIDATE_DATA_PATH = GENEREL_PATH / "validate_data"

log = get_logger()
ai_handler = AiHandler(RESULTS_PATH)
ntfy = NtfyHandler("ai_template")


def main():
    log.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    log.info(f"TF_NUM_INTRAOP_THREADS: {os.environ.get('TF_NUM_INTRAOP_THREADS')}")
    log.info(f"TF_NUM_INTEROP_THREADS: {os.environ.get('TF_NUM_INTEROP_THREADS')}")

    # with ai_handler.strategy.scope():
    time_started = 0

    try:
        time_started = ai_handler.set_time_start()

        model = defineModel()
        model.summary()

        ai_handler.plot_block_diagram(model)

        compiled_model = ai_handler.compile_model(model, metrics=["accuracy", "MeanSquaredError"])

        labeld_data = ai_handler.dataset_from_data_and_labels(
            data_dir=TRAINING_DATA_PATH / "input",
            label_dir=TRAINING_DATA_PATH / "labels",
        )
        labeld_validation = ai_handler.dataset_from_data_and_labels(
            data_dir=VALIDATE_DATA_PATH / "input",
            label_dir=VALIDATE_DATA_PATH / "labels",
        )

        # ai_handler.launch_tensorboard_threaded() # Not supported on AI-LAB
        history = ai_handler.fit_model(
            compiled_model,
            train_data=labeld_data,
            val_data=labeld_validation,
            use_tensorboard=True,
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

        ntfy.post(  # Remember the message is markdown format
            title=f"Results of ML {time_started}",
            message=(
                f"**Start time:** {time_started}\n"
                f"**Time spent:** {ai_handler.time_diff()} seconds\n\n"
                f"**Results saved to:** `{ai_handler.result_path}`\n\n"
            ),
        )

        ntfy.post_image(
            ai_handler.result_path / "model_block_diagram.png",
            title=f"Model block diagram {time_started}",
            compress=True,
        )
        ntfy.post_image(
            ai_handler.result_path / "loss.png", title="Loss diagram", compress=True
        )
        ntfy.post_image(
            ai_handler.result_path / "accuracy.png",
            title="Accuracy diagram",
            compress=True,
        )
    except Exception as e:
        ntfy.post(
            title=f"Error during model training {time_started}",
            message=f"An error occurred: {e}",
        )
        ntfy.post_image(
            ai_handler.result_path / "model_block_diagram.png",
            title=f"Model block diagram {time_started}",
            compress=True,
        )


def load_predict():
    modelPath = "results/26-09-2025_12:15:53/sum_diff_model.keras"

    model = ai_handler.load_model(modelPath)
    res = ai_handler.predict(model, [0.1, 0.3])
    print(res)


if __name__ == "__main__":
    # load_predict()
    main()
