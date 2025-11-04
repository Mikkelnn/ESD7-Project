import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, f, ncf

# Parameters
SNR_dB = 10
SNR = 10**(SNR_dB/10)
N = 16
pfa = np.linspace(0, 1, 400)

def pd_np_coherent(pfa, snr):
    return norm.sf(norm.isf(pfa) - np.sqrt(snr))

#def pd_mf(pfa, snr, mismatch=1.0):
#    return norm.sf(norm.isf(pfa) - np.sqrt(mismatch*snr))

def pd_cafar(pfa, snr, N, cfar_loss_dB=1.5):

    pd = ((1+snr)/(pfa**(-1/N)+snr))**N

    return pd

def pd_ml(pfa, snr, kappa=0.95):
    return norm.sf(norm.isf(pfa) - np.sqrt(kappa*snr))

def pd_random(pfa):
    return pfa

# Curves
Pd_np = pd_np_coherent(pfa, SNR)
#Pd_mf_ = pd_mf(pfa, SNR, mismatch=0.99)
Pd_cafar_ = pd_cafar(pfa, SNR, N, cfar_loss_dB=1.5)
Pd_ml_ = pd_ml(pfa, SNR)

pfa_target = 10e-6
pfa_target_idx = np.searchsorted(pfa, pfa_target)
pfa_target_values = [
    Pd_np[pfa_target_idx],
    pd_random(pfa)[pfa_target_idx],
    Pd_cafar_[pfa_target_idx],
    Pd_ml_[pfa_target_idx],
    1
]

font_size = 30

plt.figure(figsize=(6, 6))

# Plot ROC curves
plt.plot(pfa, Pd_np, 'b-', label='NP')
plt.plot(pfa, pd_random(pfa), 'y-', label='Random')
plt.plot(pfa, Pd_cafar_, 'r-', label='CA-CFAR')
plt.plot(pfa, Pd_ml_, 'g-', label='ML')

# Mark P_FA target
plt.axvline(x=pfa_target, color='magenta', linestyle=':', linewidth=2, label=f'P_FA = {pfa_target:.0e}')
plt.text(pfa_target, 0.02, f'P_FA = {pfa_target:.0e}', color='magenta', rotation=90, va='bottom', ha='right', fontsize=font_size)

# Crosses and horizontal lines at P_FA=1e-6 for each detector
cross_info = [
    ('blue', 'NP', Pd_np[pfa_target_idx]),
    ('red', 'CA-CFAR', Pd_cafar_[pfa_target_idx]),
    ('green', 'ML', Pd_ml_[pfa_target_idx])
]

for color, label, value in cross_info:
    plt.scatter(pfa_target, value, color=color, s=60, marker='x', zorder=5)
    plt.hlines(value, xmin=0, xmax=1, colors=color, linestyles=':', linewidth=1.2, label=f'{label} P_D={value:.3g}')

plt.xlabel('P_FA', fontsize=font_size)
plt.ylabel('P_D', fontsize=font_size)
plt.title(f'ROC Curves (SNR = {SNR_dB:.1f} dB)', fontsize=font_size)
plt.ylim(0, 1.05)
plt.xlim(-0.05, 1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='lower right', fontsize=font_size*0.9)
plt.tick_params(axis='both', which='major', labelsize=font_size)
plt.tight_layout()
plt.show()
