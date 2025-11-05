import numpy as np
import math

def radar_received_power(Pt, Gt, Gr, wavelength, sigma, R):
    """Calculate received power using the radar range equation."""
    return (Pt * Gt * Gr * wavelength**2 * sigma) / ((4 * math.pi)**3 * R**4)

# Example usage:
Pt = 10         # 10 W transmit power
Gt = 12.5       # (10.97 dBi) transmit gain (linear)
Gr = Gt         # receive gain (linear)
freq = 75e9     # 75 GHz
c = 3e8         # speed of light (m/s)
wavelength = c / freq
sigma = 3.47 * 10**(-6) # 1 m² RCS
R = 1e3          # 1 km

Pr = radar_received_power(Pt, Gt, Gr, wavelength, sigma, R)
dbm_r = 10 * np.log10(Pr) + 30
print(f"Received Power: {dbm_r} dbm")
