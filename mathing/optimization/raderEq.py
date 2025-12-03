# example: elementwise -> returns (2,1)
import numpy as np
from scipy.constants import k

Pt=10; Gt=12.5; Gr=Gt; c=3e8; freq=75e9
wavelength = c/freq
const = Pt * Gt * Gr * wavelength**2 / ((4*np.pi)**3)

# to get 4 values (2x2) use broadcasting or outer product
sigma = np.array([3.47e-6 , 0.035])[:, None]   # shape (2,1)
R     = np.array([100.0, 2000.0])[None, :]  # shape (1,2)
Pr = const * sigma / (R**4)  # shape (2,2) -> 4 combinations

PrdBm = 10 * np.log10(Pr) + 30

print(f"Received power (dBm): {PrdBm}")

B = 20e6

T_low = 253.15
T_high = 353.15
N_low = k * T_low * B       # W
N_high = k * T_high * B     # W
N_low_dBm = 10 * np.log10(N_low) + 30
N_high_dBm = 10 * np.log10(N_high) + 30
print(f"Noise floor at {T_low} K: {N_low_dBm} dBm")
print(f"Noise floor at {T_high} K: {N_high_dBm} dBm")