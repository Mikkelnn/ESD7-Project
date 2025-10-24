# New script calculating parameters for the stochastic analysis of how much trash we expect to detect in LEO
# Debris is assumed to be uniformly distributed throughout LEO
# ALL UNITS ARE SI!!!!!!!

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

r_Earth = 6371e3
M_Earth = 5.97219e24
Grav_Const = 6.6743e-11
r_lower = 180e3 + r_Earth
r_upper = 2000e3 + r_Earth
V_LEO = (4.0/3.0) * np.pi * (np.power(r_upper, 3) - np.power(r_lower, 3))

trash_LEO = 934000
density = trash_LEO / V_LEO

ant_range = 1000.0
ant_orbit = r_Earth + 550e3
theta_HPBW = np.deg2rad(27.0)
phi_HPBW   = np.deg2rad(88.0)


def f1Integral(r):
    b = -np.tan(theta_HPBW/2) * ant_orbit
    result = (np.tan(theta_HPBW/2)/3) * r**3 + (b * r**2)/2
    return result


def singleImage():
    V_ant = (np.pi / 3.0) * (ant_range ** 3) * np.tan(theta_HPBW/2) * np.tan(phi_HPBW/2)   # correct truncated-cone volume
    E_trashImage = density * V_ant
    print(f"Single image expected debris: {E_trashImage:.4e}")
    return E_trashImage


def plot_single_orbit_geometry(ant_orbit, ant_range, theta_HPBW):
    """
    Creates a 3D visualization of the antenna beam volume for ±10° azimuth,
    mirrored about the XY plane.
    """
    b = -np.tan(theta_HPBW/2) * ant_orbit
    r2 = ant_orbit + np.cos(theta_HPBW/2) * ant_range
    r3 = ant_orbit + ant_range

    # Define the two radial functions
    f1 = lambda r: np.tan(theta_HPBW/2)*r + b
    f2 = lambda r: np.sqrt(np.maximum(ant_range**2 - (r - ant_orbit)**2, 0))

    # Sweep ±10° in azimuth
    phi_range = np.radians(2)
    phi = np.linspace(-phi_range/2, phi_range/2, 100)

    # Region 1 and Region 2 (meshgrids)
    r1_vals = np.linspace(ant_orbit, r2, 200)
    r2_vals = np.linspace(r2, r3, 200)
    R1, PHI1 = np.meshgrid(r1_vals, phi)
    R2, PHI2 = np.meshgrid(r2_vals, phi)

    # Compute heights and convert to Cartesian
    Z1 = f1(R1)
    Z2 = f2(R2)
    X1 = R1 * np.cos(PHI1)
    Y1 = R1 * np.sin(PHI1)
    X2 = R2 * np.cos(PHI2)
    Y2 = R2 * np.sin(PHI2)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Upper half
    ax.plot_surface(X1, Y1, Z1, color='dodgerblue', alpha=0.6, rstride=3, cstride=3)
    ax.plot_surface(X2, Y2, Z2, color='deepskyblue', alpha=0.6, rstride=3, cstride=3)

    # Lower half (mirror)
    ax.plot_surface(X1, Y1, -Z1, color='tomato', alpha=0.4, rstride=3, cstride=3)
    ax.plot_surface(X2, Y2, -Z2, color='lightcoral', alpha=0.4, rstride=3, cstride=3)

    # Mark transition point (r2)
    ax.plot([r2*np.cos(0)], [0], [0], 'ko', markersize=6, label='Transition r2')

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Antenna Beam Volume (±10° Azimuth)")
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 0.5])
    ax.legend()

    plt.show()


def singleOrbit():
    b = -np.tan(theta_HPBW/2) * ant_orbit
    f1 = lambda r: (np.tan(theta_HPBW/2)*r + b) * r
    r2 = ant_orbit + np.cos(theta_HPBW/2) * ant_range
    r3 = ant_orbit + ant_range
    f2 = lambda r: (np.sqrt(ant_range**2 - (r - ant_orbit)**2)) * r

    I1, _ = integrate.quad(f1, ant_orbit, r2)
    I2, _ = integrate.quad(f2, r2, r3)

    Volume = 4 * np.pi * (I1 + I2)
    ESingleOrbit = density * Volume
    print(f"Single orbit volume: {Volume:.4e} m³, expected debris: {ESingleOrbit}")

    # Call the 3D visualization
    # plot_single_orbit_geometry(ant_orbit, ant_range, theta_HPBW)

    return ESingleOrbit

def orbitTime(R):
    radiusConst = (R**3) / (Grav_Const * M_Earth)
    orbitTime = 2 * np.pi * np.sqrt(radiusConst)
    return orbitTime


def movingDebris(R, tilt, rotation):
    debrisRange = ant_orbit + R
    cosTilt = np.cos(np.rad2deg(tilt))
    sinTilt = np.sin(np.rad2deg(tilt))
    cosRot = np.cos(np.rad2deg(rotation))
    sinRot = np.sin(np.rad2deg(rotation))

    T = orbitTime(debrisRange)

    Xt = lambda t: debrisRange * (cosRot * np.cos(t/T) + sinTilt * sinRot * np.sin(t/T))
    Yt = lambda t: debrisRange * (sinRot * np.cos(t/T) + sinTilt * cosRot * np.sin(t/T))
    Zt = lambda t: debrisRange * cosTilt * np.sin(t/T)


def main():
    E_TrashSingle = singleImage()
    ESingleOrbit = singleOrbit()



if __name__ == "__main__":
    main()
