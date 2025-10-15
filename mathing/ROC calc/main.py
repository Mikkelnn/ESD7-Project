import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, f, ncf

# Parameters
SNR_dB = 10.0
SNR = 10**(SNR_dB/10)
N = 16
pfa = np.logspace(-10, 1, 400)

def pd_np_coherent(pfa, snr):
    return norm.sf(norm.isf(pfa) - np.sqrt(snr))

def pd_mf(pfa, snr, mismatch=1.0):
    return norm.sf(norm.isf(pfa) - np.sqrt(mismatch*snr))

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

# Curves
Pd_np = pd_np_coherent(pfa, SNR)
Pd_mf_ = pd_mf(pfa, SNR, mismatch=0.99)
Pd_cafar_ = pd_cafar(pfa, SNR, N, cfar_loss_dB=1.5)
Pd_ml_ = pd_ml(pfa, SNR)

# Plot
plt.figure(figsize=(8,6))
plt.semilogx(pfa, Pd_np, 'b-',  label='NP')
plt.semilogx(pfa, Pd_mf_, 'm-', label='Matched Filter')
plt.semilogx(pfa, Pd_cafar_, 'r',  label='CA-CFAR')
plt.semilogx(pfa, Pd_ml_, 'g',  label='Super cool Machine learning Benny Algorithm')
plt.xlabel('P_FA')
plt.ylabel('P_D')
plt.title(f'ROC (SNR = {SNR_dB} dB)')
plt.ylim(0, 1)
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
