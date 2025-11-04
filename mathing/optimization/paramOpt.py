import numpy as np
import sympy as sp
from scipy.optimize import minimize

# ------------------------
# Constants
# ------------------------
c = 3e8                    # speed of light [m/s]
sigma = 1e-2               # radar cross-section [m^2]
G_dBi = 10.97              # gain [dBi]
G = 10 ** (G_dBi / 10)     # linear gain
Nc = 128                   # number of chirps

# variable bounds
f0_bounds = (5e9, 100e9)
B_bounds  = (10e6, 4e9)
Tc_bounds = (1e-6, 50e-6)

# ------------------------
# Symbolic setup (for KKT)
# ------------------------
f0, B, Tc = sp.symbols('f0 B Tc', positive=True)

lam_expr = c / f0
fs_expr = 2.2 * B  # f_s = 2B + B/5

# Radar equations
R_rel = (G**2 * lam_expr**2 * sigma / B)**(1/4)     # relative detection range
Rmax_FMCW = c * fs_expr * Tc / (4 * B)              # unambiguous range for FMCW
Vr_max = c / (4 * f0 * Tc)
dR = c / (2 * B)
dVr = c / (2 * f0 * Nc * Tc)

# Multi-objective scalarization (tunable weights)
alpha, beta, gamma, delta, eta = 1, 1.5, 1, 1, 0.5  # weights
obj = -(alpha * R_rel + beta * Vr_max + eta * Rmax_FMCW) + (gamma * dR + delta * dVr)

# Lagrangian for KKT
l1, l2, l3, l4, l5, l6 = sp.symbols('l1 l2 l3 l4 l5 l6', real=True, nonnegative=True)
constraints = [
    f0 - f0_bounds[0],
    f0_bounds[1] - f0,
    B - B_bounds[0],
    B_bounds[1] - B,
    Tc - Tc_bounds[0],
    Tc_bounds[1] - Tc
]
L = obj - sum([l * con for l, con in zip([l1, l2, l3, l4, l5, l6], constraints)])
gradL = [sp.diff(L, v) for v in [f0, B, Tc]]

print("Symbolic KKT Stationarity Conditions:")
for g in gradL:
    sp.pprint(sp.simplify(g))
    print()

# ------------------------
# Numerical optimization
# ------------------------
def objective(vars):
    f0, B, Tc = vars
    lam = c / f0
    fs = 2.2 * B
    R_rel = (G**2 * lam**2 * sigma / B)**0.25
    Rmax_FMCW = c * fs * Tc / (4 * B)
    Vr_max = c / (4 * f0 * Tc)
    dR = c / (2 * B)
    dVr = c / (2 * f0 * Nc * Tc)
    # scalarized objective
    return -(alpha * R_rel + beta * Vr_max + eta * Rmax_FMCW) + (gamma * dR + delta * dVr)

bounds = [f0_bounds, B_bounds, Tc_bounds]
x0 = [5e9, 1e8, 1e-5]

res = minimize(objective, x0, bounds=bounds, method='SLSQP')

print("\nOptimized values:")
print(f"f0 = {res.x[0]/1e9:.3f} GHz")
print(f"B  = {res.x[1]/1e6:.3f} MHz")
print(f"Tc = {res.x[2]*1e6:.3f} µs")
print(f"Objective = {res.fun:.4e}")

# Derived metrics
f0_opt, B_opt, Tc_opt = res.x
fs_opt = 2.2 * B_opt
R_rel_opt = (G**2 * (c/f0_opt)**2 * sigma / B_opt)**0.25
Rmax_FMCW_opt = c * fs_opt * Tc_opt / (4 * B_opt)
Vr_max_opt = c / (4 * f0_opt * Tc_opt)
dR_opt = c / (2 * B_opt)
dVr_opt = c / (2 * f0_opt * Nc * Tc_opt)

print("\nDerived performance metrics:")
print(f"R_rel = {R_rel_opt:.4e} (relative)")
print(f"R_max,FMCW = {Rmax_FMCW_opt:.2f} m")
print(f"V_r,max = {Vr_max_opt:.2f} m/s")
print(f"ΔR = {dR_opt:.4f} m")
print(f"ΔV_r = {dVr_opt:.4f} m/s")
print(f"f_s = {fs_opt/1e6:.2f} MHz")
