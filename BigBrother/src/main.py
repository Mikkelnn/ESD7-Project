from ai_handler import AiHandler
from ntfy import NtfyHandler
from logger import get_logger
import matplotlib.pyplot as plt
from pathlib import Path

log = get_logger()

ai_handler = AiHandler()
ntfy = NtfyHandler("ai_template")

RESULTS_PATH = Path("results/")

def main():
    time_started = ai_handler.set_time_start()

    # Save results to a folder named after time_started
    result_folder = RESULTS_PATH / str(time_started)
    result_folder.mkdir(parents=True, exist_ok=True)

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
       title="Results of ai_template",
       message=(
           f"**Start time:** {time_started}\n"
           f"**Time spent:** {ai_handler.time_diff()} seconds\n\n"
           f"**Results saved to:** `{result_folder}`\n\n"
       )
    )

    ntfy.post_image(result_folder / "results.png", title="Linear Regression Results", compress=False)

if __name__ == "__main__":
    main()