import numpy as np
from scipy.optimize import minimize

# --- constants ---
c = 3e8
sigma = 1e-2
G_dBi = 10.97
G = 10 ** (G_dBi / 10)
Nc = 128
k_B = 1.380649e-23  # Boltzmann constant [J/K]

# variable bounds
f0_bounds = (5e9, 100e9)
B_bounds  = (10e6, 4e9)
Tc_bounds = (1e-6, 50e-6)

# performance constraints (example values)
RmaxFMCW_min = 700.0      # m
RmaxFMCW_max = 50000.0
Vrmax_min = 1500.0          # m/s
Vrmax_max = 30000.0
dR_max = 10.0              # m
dVr_max = 50.0            # m/s

# weights
alpha, beta, gamma, delta, eta = 1, 1, 1, 1, 1

# -----------------------------------------------------
# Metric computation function
# -----------------------------------------------------
def metrics(vars):
    f0, B, Tc = vars
    lam = c / f0
    fs = 2.2 * B
    R_rel = (G**2 * lam**2 * sigma / B)**0.25
    Rmax_FMCW = c * fs * Tc / (4 * B)
    Vr_max = c / (4 * f0 * Tc)
    dR = c / (2 * B)
    dVr = c / (2 * f0 * Nc * Tc)
    
    # noise floor at -20°C (253.15 K) and +80°C (353.15 K)
    T_low = 253.15
    T_high = 353.15
    N_low = k_B * T_low * B       # W
    N_high = k_B * T_high * B     # W
    N_low_dBm = 10*np.log10(N_low) + 30
    N_high_dBm = 10*np.log10(N_high) + 30
    
    return R_rel, Rmax_FMCW, Vr_max, dR, dVr, N_low_dBm, N_high_dBm

# -----------------------------------------------------
# Objective
# -----------------------------------------------------
def objective(vars):
    R_rel, Rmax_FMCW, Vr_max, dR, dVr, _, _ = metrics(vars)
    return -(alpha * R_rel + beta * Vr_max + eta * Rmax_FMCW) + (gamma * dR + delta * dVr)

# -----------------------------------------------------
# Constraints
# -----------------------------------------------------
constraints = [
    {'type': 'ineq', 'fun': lambda x: metrics(x)[1] - RmaxFMCW_min},
    {'type': 'ineq', 'fun': lambda x: RmaxFMCW_max - metrics(x)[1]},
    {'type': 'ineq', 'fun': lambda x: metrics(x)[2] - Vrmax_min},
    {'type': 'ineq', 'fun': lambda x: Vrmax_max - metrics(x)[2]},
    {'type': 'ineq', 'fun': lambda x: dR_max - metrics(x)[3]},
    {'type': 'ineq', 'fun': lambda x: dVr_max - metrics(x)[4]},
]

bounds = [f0_bounds, B_bounds, Tc_bounds]
x0 = [5e9, 1e8, 10e-6]

res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')

# -----------------------------------------------------
# Output Results
# -----------------------------------------------------
f0_opt, B_opt, Tc_opt = res.x
R_rel, RmaxFMCW, Vr_max, dR, dVr, N_low_dBm, N_high_dBm = metrics(res.x)
fs_opt = 2.2 * B_opt

NoiseFigure = 12 #See page 39 here: https://www.ti.com/lit/ds/symlink/awr2944.pdf?ts=1741521817308&ref_url=https%253A%252F%252Fwww.ti.com%252Fproduct%252FAWR2944
SNRin = [0, 5, 10]


print("\nOptimized variables:")
print(f"f0 = {f0_opt/1e9:.3f} GHz")
print(f"B  = {B_opt/1e6:.3f} MHz")
print(f"Tc = {Tc_opt*1e6:.3f} µs")

print("\nPerformance metrics:")
print(f"R_rel (relative) = {R_rel:.4e}")
print(f"R_max,FMCW = {RmaxFMCW:.2f} m")
print(f"V_r,max = {Vr_max:.2f} m/s")
print(f"ΔR = {dR:.3f} m")
print(f"ΔV_r = {dVr:.3f} m/s")

print("\nNoise floor levels:")
print(f"At -20 °C: {N_low_dBm:.2f} dBm")
print(f"At +80 °C: {N_high_dBm:.2f} dBm")

print (f"\nPmin for highest noise floor with a noisefigure of {NoiseFigure} dB")
print(f"At SNRin = {SNRin[0]}dB: {N_high_dBm + NoiseFigure + SNRin[0]}")
print(f"At SNRin = {SNRin[1]}dB: {N_high_dBm + NoiseFigure + SNRin[1]}")
print(f"At SNRin = {SNRin[2]}dB: {N_high_dBm + NoiseFigure + SNRin[2]}")

print(f"\nSampling rate f_s = {fs_opt/1e6:.2f} MHz")
