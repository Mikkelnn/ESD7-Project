from ai_handler import AiHandler
from ntfy import NtfyHandler
from logger import get_logger
from pathlib import Path
from model import defineModel 

log = get_logger()

RESULTS_PATH = Path("results/")
ai_handler = AiHandler(RESULTS_PATH)
ntfy = NtfyHandler("ai_template")

TRAINING_DATA_PATH = Path("data/")
VALIDATE_DATA_PATH = Path("validate/")


def main():
    ai_handler.set_time_start()

    model = defineModel()

    ai_handler.print_summary(model)
    ai_handler.plot_block_diagram(model)

    compiled_model = ai_handler.compile_model(model)

        labeld_data = ai_handler.dataset_from_data_and_labels(data_dir=TRAINING_DATA_PATH / "data", label_dir=TRAINING_DATA_PATH / "label")
        labeld_validation = ai_handler.dataset_from_data_and_labels(data_dir=VALIDATE_DATA_PATH / "data", label_dir=VALIDATE_DATA_PATH / "label")

        ai_handler.launch_tensorboard_threaded()
        history = ai_handler.fit_model(compiled_model, train_data=labeld_data, val_data=labeld_validation, use_tensorboard=True)
    except Exception as e:
        ntfy.post(
            title=f"Error during model training {time_started}",
            message=f"An error occurred: {e}"
        )
        ntfy.post_image(ai_handler.result_path / "model_block_diagram.png", title=f"Model block diagram {time_started}", compress=True)
        raise e

    ai_handler.save_model(model, name=f"sum_diff_model")

    ai_handler.wait_for_ctrl_c()

    # Plot and save the figure using matplotlib
    #plt.figure()
    #plt.scatter(x, y, label="True data")
    #plt.plot(x, y_pred, color="red", label="Model prediction")
    #plt.legend()
    #plt.title("Linear Regression: True vs Predicted")
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.savefig(result_folder / "results.svg", format="svg")
    #plt.savefig(result_folder / "results.png", format="png")
    #plt.close()

    ai_handler.set_time_stop()

    ntfy.post(  # Remember the message is markdown format
       title=f"Results of ML {time_started}",
       message=(
           f"**Start time:** {time_started}\n"
           f"**Time spent:** {ai_handler.time_diff()} seconds\n\n"
           f"**Results saved to:** `{ai_handler.result_path}`\n\n"
       )
    )

    ntfy.post_image(ai_handler.result_path / "model_block_diagram.png", title=f"Model block diagram {time_started}", compress=True)

def load_predict():
    modelPath = 'results/26-09-2025_12:15:53/sum_diff_model.keras'

    model = ai_handler.load_model(modelPath)
    res = ai_handler.predict(model, [0.1, 0.3])
    print(res)

if __name__ == "__main__":
    load_predict()
    # main()