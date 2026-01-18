import numpy as np
import re
from pathlib import Path

file_path = "sweep_localization_data.txt"

# Lists to store each column separately
range_pred_raw = []
range_label_raw = []
velocity_pred_raw = []
velocity_label_raw = []

range_pred_scaled = []
range_label_scaled = []
velocity_pred_scaled = []
velocity_label_scaled = []

# Parse file
with open(file_path, "r") as f:
    for line in f:
        pred_raw = re.findall(r'predicted_raw:\s*\[\[([0-9.\s-]+)\]\]', line)
        label_raw = re.findall(r'label_raw:\s*\[([0-9.\s-]+)\]', line)
        pred_scaled = re.findall(r'predicted_raw_scaled:\s*\[\[([0-9.\s-]+)\]\]', line)
        label_scaled = re.findall(r'label_raw_scaled:\s*\[([0-9.\s-]+)\]', line)

        if pred_raw and label_raw and pred_scaled and label_scaled:
            pred_raw_vals = [float(x) for x in pred_raw[0].split()]
            label_raw_vals = [float(x) for x in label_raw[0].split()]
            pred_scaled_vals = [float(x) for x in pred_scaled[0].split()]
            label_scaled_vals = [float(x) for x in label_scaled[0].split()]

            # Separate range (index 0) and velocity (index 1)
            range_pred_raw.append(pred_raw_vals[0])
            range_label_raw.append(label_raw_vals[0])
            velocity_pred_raw.append(pred_raw_vals[1])
            velocity_label_raw.append(label_raw_vals[1])

            range_pred_scaled.append(pred_scaled_vals[0])
            range_label_scaled.append(label_scaled_vals[0])
            velocity_pred_scaled.append(pred_scaled_vals[1])
            velocity_label_scaled.append(label_scaled_vals[1])

# Convert to numpy arrays
range_pred_raw = np.array(range_pred_raw)
range_label_raw = np.array(range_label_raw)
velocity_pred_raw = np.array(velocity_pred_raw)
velocity_label_raw = np.array(velocity_label_raw)

range_pred_scaled = np.array(range_pred_scaled)
range_label_scaled = np.array(range_label_scaled)
velocity_pred_scaled = np.array(velocity_pred_scaled)
velocity_label_scaled = np.array(velocity_label_scaled)

def compute_metrics(pred, label, units, name="Data"):
    abs_error = np.abs(pred - label)
    rel_error = abs_error / np.maximum(np.abs(label), 1e-8) * 100  # relative error %

    percentiles = [1, 5, 10, 20, 50, 75, 90, 95, 98, 100]
    perc_values_units = np.percentile(abs_error, percentiles)  # report in units

    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((pred - label)**2))

    print(f"\n{name} Metrics ({units}):")
    print(f"  MAE: {mae:.4f} {units}")
    print(f"  RMSE: {rmse:.4f} {units}")
    print("  Error Percentiles:")
    for p, val in zip(percentiles, perc_values_units):
        print(f"    {p}th percentile: {val:.4f} {units}")
    
    # Tolerance coverage
    for key in [1, 5, 10, 20, 50, 75, 90, 95, 98, 100]:
        tol_level = np.percentile(rel_error, key)
        print(f"    Tolerance covering {key}% of predictions: ±{tol_level:.2f}%")

# Compute metrics for Range
# compute_metrics(range_pred_raw, range_label_raw, units="m", name="Raw Range")
compute_metrics(range_pred_scaled, range_label_scaled, units="m", name="Scaled Range")

# Compute metrics for Velocity
# compute_metrics(velocity_pred_raw, velocity_label_raw, units="m/s", name="Raw Velocity")
compute_metrics(velocity_pred_scaled, velocity_label_scaled, units="m/s", name="Scaled Velocity")
