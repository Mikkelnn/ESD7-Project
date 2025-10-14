"""
ROC Curve Analysis for Radar Detection Algorithms
Compares Neyman-Pearson, CA-CFAR, and Machine Learning detection performance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, f, ncf
from sklearn.metrics import auc

# Radar system parameters
snr_db = 5  # Signal-to-noise ratio in dB
snr_linear = 10**(snr_db / 10)  # SNR in linear scale
num_reference_cells = 64  # Number of reference cells for CFAR
false_alarm_rates = np.logspace(-6, -0.3, 1000)  # P_fa from 10^-6 to ~0.5

# Numerical stability parameters
pfa_clip_min = 1e-15  # Minimum P_fa value to avoid numerical issues
pfa_clip_max = 1 - pfa_clip_min  # Maximum P_fa value to avoid numerical issues

def calculate_neyman_pearson_pd(false_alarm_rates, snr_linear):
    """
    Calculate probability of detection for Neyman-Pearson detector.
    
    Args:
        false_alarm_rates: Array of false alarm probabilities (P_fa)
        snr_linear: Signal-to-noise ratio in linear scale
    
    Returns:
        Array of detection probabilities (P_d)
    """
    # Clip P_fa to valid range [0, 1] to avoid numerical issues
    pfa_clipped = np.clip(false_alarm_rates, pfa_clip_min, pfa_clip_max)
    detection_rates = norm.sf(norm.isf(pfa_clipped) - np.sqrt(snr_linear))
    return detection_rates


def calculate_ca_cfar_pd(false_alarm_rates, snr_linear, num_ref_cells, cfar_loss_db=0):
    """
    Calculate probability of detection for CA-CFAR detector.
    
    Args:
        false_alarm_rates: Array of false alarm probabilities (P_fa)
        snr_linear: Signal-to-noise ratio in linear scale
        num_ref_cells: Number of reference cells for CFAR processing
        cfar_loss_db: CFAR implementation loss in dB (typically 1-2 dB)
    
    Returns:
        Array of detection probabilities (P_d)
    """
    # Effective SNR accounting for CFAR loss
    effective_snr = snr_linear / (10**(cfar_loss_db / 10))
    
    # Degrees of freedom for F-distribution
    degrees_numerator = 2
    degrees_denominator = 2 * num_ref_cells

    # Clip P_fa to valid range to avoid numerical issues
    pfa_clipped = np.clip(false_alarm_rates, pfa_clip_min, pfa_clip_max)
    
    # Calculate threshold from false alarm rate
    threshold = f.isf(pfa_clipped, degrees_numerator, degrees_denominator)

    # Calculate detection probability using non-central F-distribution
    detection_rates = ncf.sf(threshold, degrees_numerator, degrees_denominator, 
                            2 * effective_snr)
    return detection_rates


def calculate_ml_pd(false_alarm_rates, snr_linear, efficiency_factor=0.85):
    """
    Calculate probability of detection for Machine Learning detector.
    
    Args:
        false_alarm_rates: Array of false alarm probabilities (P_fa)
        snr_linear: Signal-to-noise ratio in linear scale
        efficiency_factor: ML algorithm efficiency (0 < kappa <= 1)
    
    Returns:
        Array of detection probabilities (P_d)
    """
    # Clip P_fa to valid range [0, 1] to avoid numerical issues
    pfa_clipped = np.clip(false_alarm_rates, pfa_clip_min, pfa_clip_max)
    detection_rates = norm.sf(norm.isf(pfa_clipped) - np.sqrt(efficiency_factor * snr_linear))
    return detection_rates

def plot_roc_curves():
    """Generate and display ROC curves for different detection algorithms."""
    
    # Calculate detection probabilities for each algorithm
    pd_neyman_pearson = calculate_neyman_pearson_pd(false_alarm_rates, snr_linear)
    pd_ca_cfar = calculate_ca_cfar_pd(false_alarm_rates, snr_linear, 
                                      num_reference_cells, cfar_loss_db=2)
    pd_machine_learning = calculate_ml_pd(false_alarm_rates, snr_linear)

    # Calculate standard ROC Area Under Curve (AUC)
    # Sort arrays by P_fa for proper numerical integration
    sorted_indices = np.argsort(false_alarm_rates)
    pfa_sorted = false_alarm_rates[sorted_indices]
    pd_np_sorted = pd_neyman_pearson[sorted_indices]
    pd_cfar_sorted = pd_ca_cfar[sorted_indices]
    pd_ml_sorted = pd_machine_learning[sorted_indices]
    
    # Normalize to [0,1] range for standard ROC AUC
    pfa_normalized = (pfa_sorted - pfa_sorted.min()) / (pfa_sorted.max() - pfa_sorted.min())
    auc_neyman_pearson = auc(pfa_normalized, pd_np_sorted)
    auc_ca_cfar = auc(pfa_normalized, pd_cfar_sorted) 
    auc_machine_learning = auc(pfa_normalized, pd_ml_sorted)

    # Print performance metrics
    print("ROC Performance Analysis")
    print("=" * 40)
    print(f"SNR: {snr_db} dB")
    print(f"Reference cells (CFAR): {num_reference_cells}")
    print()
    print("Standard ROC AUC (0-1 scale):")
    print(f"  Neyman-Pearson:   {auc_neyman_pearson:.4f}")
    print(f"  Machine Learning: {auc_machine_learning:.4f}")
    print(f"  CA-CFAR:          {auc_ca_cfar:.4f}")

    # Create ROC curve plot
    plt.figure(figsize=(10, 7))
    plt.plot(false_alarm_rates, pd_neyman_pearson, 'b-', linewidth=2,
             label=f'Neyman-Pearson (AUC={auc_neyman_pearson:.4f})')
    plt.plot(false_alarm_rates, pd_machine_learning, 'g:', linewidth=2,
             label=f'Machine Learning (AUC={auc_machine_learning:.4f})')
    plt.plot(false_alarm_rates, pd_ca_cfar, 'r--', linewidth=2,
             label=f'CA-CFAR (AUC={auc_ca_cfar:.4f})')
    
    
    # Format plot with actual P_fa values
    plt.xlabel('Probability of False Alarm (P_FA)', fontsize=12)
    plt.ylabel('Probability of Detection (P_D)', fontsize=12)
    plt.title(f'ROC Curves Comparison (SNR = {snr_db} dB)', fontsize=14)
    
    # Use the actual false_alarm_rates values on x-axis
    plt.xlim(false_alarm_rates.min(), false_alarm_rates.max())
    plt.ylim(min(pd_neyman_pearson.min(), pd_ca_cfar.min(), pd_machine_learning.min()), 1.0)
    
    # Force scientific notation on x-axis
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_roc_curves()
