import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, f, ncf

# Parameters
SNR_dB = 12.0
SNR = 10**(SNR_dB/10)
N = 16
pfa = np.linspace(0, 1, 400)

def pd_np_coherent(pfa, snr):
    return norm.sf(norm.isf(pfa) - np.sqrt(snr))

#def pd_mf(pfa, snr, mismatch=1.0):
#    return norm.sf(norm.isf(pfa) - np.sqrt(mismatch*snr))

def pd_cafar(pfa, snr, N, cfar_loss_dB=1.5):
    rho_eff = snr / (10**(cfar_loss_dB/10))  # optional CFAR loss (1–2 dB typical)
    dfn, dfd = 2, 2*N

    # Threshold tau from P_FA using central F quantile
    tau = f.isf(pfa, dfn, dfd)  # inverse survival = 1 - CDF

    # P_D from noncentral F survival at same threshold
    pd = ncf.sf(tau, dfn, dfd, 2*rho_eff)
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

# Plot
plt.figure(figsize=(8,6))
plt.plot(pfa, Pd_np, 'b-',  label='NP')
plt.plot(pfa, pd_random(pfa), 'k--', label='Random')
#plt.plot(pfa, Pd_mf_, 'm-', label='Matched')
plt.plot(pfa, Pd_cafar_, 'r',  label='CA-CFAR')
plt.plot(pfa, Pd_ml_, 'g',  label='ML')
plt.scatter(0, 1, color='blue', s=80, label='Perfect')  # Add point at (0,1)
plt.axvline(x=1e-6, color='magenta', linestyle=':', linewidth=2, label='P_FA target = 1e-6')  # Vertical marker
plt.text(1e-6, 0.02, 'P_FA=1e-6', color='magenta', rotation=90, va='bottom', ha='right', fontsize=10)

# Crosses and horizontal lines at P_FA=1e-6 for each detector
cross_colors = ['blue', 'black', 'red', 'green', 'blue']
cross_labels = ['NP', 'Random', 'CA-CFAR', 'ML', 'Perfect']
cross_values = [
    Pd_np[np.searchsorted(pfa, 1e-6)],
    pd_random(pfa)[np.searchsorted(pfa, 1e-6)],
    Pd_cafar_[np.searchsorted(pfa, 1e-6)],
    Pd_ml_[np.searchsorted(pfa, 1e-6)],
    1
]

for color, label, value in zip(cross_colors, cross_labels, cross_values):
    plt.scatter(1e-6, value, color=color, s=60, marker='x', label=f'{label} @ P_FA=1e-6')
    plt.hlines(value, xmin=0, xmax=1, colors=color, linestyles=':', linewidth=1.5, label=f'{label} P_D={value:.3g}')

plt.xlabel('P_FA')
plt.ylabel('P_D')
plt.title('ROC Curves')
plt.ylim(0, 1.05)
plt.xlim(-0.05, 1)
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
