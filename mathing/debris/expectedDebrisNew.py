# New script calculating parameters for the stochastic analysis of how much trash we expect to detect in LEO
# Debris is assumed to be uniformly distributed throughout LEO
# ALL UNITS ARE SI!!!!!!!

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from fractions import Fraction

r_Earth = 6371e3
M_Earth = 5.97219e24
Grav_Const = 6.6743e-11
r_lower = 180e3 + r_Earth
r_upper = 2000e3 + r_Earth
V_LEO = (4.0/3.0) * np.pi * (np.power(r_upper, 3) - np.power(r_lower, 3))

trash_LEO = 1036500
density = trash_LEO / V_LEO

ant_range = 1000.0
ant_orbit = r_Earth + 550e3
theta_HPBW = np.deg2rad(27.0)
phi_HPBW   = np.deg2rad(88.0)
ant_orbit_time = 2 * np.pi * np.sqrt((ant_orbit**3)/(Grav_Const * M_Earth))


def f1Integral(r):
    b = -np.tan(theta_HPBW/2) * ant_orbit
    result = (np.tan(theta_HPBW/2)/3) * r**3 + (b * r**2)/2
    return result


def singleImage():
    print(ant_orbit_time)
    print(f"V_LEO = {V_LEO} Debris density = {density}")
    V_ant = (np.pi / 3.0) * (ant_range ** 3) * np.tan(theta_HPBW/2) * np.tan(phi_HPBW/2)   # correct truncated-cone volume
    E_trashImage = density * V_ant
    print(f"Single image expected debris: {E_trashImage:.4e} Antenna volume: {V_ant}")
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

def archAngle(R):
    b = np.tan(theta_HPBW/2) * R
    a = np.tan(phi_HPBW/2) * R
    c = np.sqrt((a/2)**2 - b**2)
    orbitAngle = c/(ant_orbit + R)
    return orbitAngle

def common_alignment_time(T1, T2, max_denominator=5000, tol=1e-10):
    """
    Finds the smallest time where two periods T1 and T2 align again
    (i.e., n*T1 = m*T2 for integers n,m).

    Returns (t_common, info) where:
      - t_common: float or None if no close match
      - info: dict with m, n, approx_error, ratio
    """
    ratio = T1 / T2
    frac = Fraction(ratio).limit_denominator(max_denominator)
    m, n = frac.numerator, frac.denominator
    approx = m / n
    error = abs(ratio - approx) / abs(ratio)

    t_common = n * T1  # n*T1 = m*T2 approximately
    info = {"m": m, "n": n, "approx_error": error, "ratio": ratio, "frac": frac}

    if error <= tol:
        return t_common, info
    else:
        return None, info


# -------------------------------
# Helper: fraction of time debris is inside ellipse
# -------------------------------
def fraction_inside_ellipse(debrisPos, antPos, T_debris, T_ant, a,
                            num_points=20000, max_denominator=5000, tol=1e-10,
                            num_orbits_if_ratio1=10000):
    """
    Computes the fraction of time the debris is inside the ellipse.
    If the period ratio is ~1, simulate many orbits.
    """
    t_common, info = common_alignment_time(T_debris, T_ant,
                                           max_denominator=max_denominator,
                                           tol=tol)

    # Check if ratio ~ 1
    if abs(info['ratio'] - 1.0) <= 1e-4:
        # Use many orbits
        t_total = T_debris * num_orbits_if_ratio1
        print(f"⚠ Period ratio ≈ 1. Simulating {num_orbits_if_ratio1} debris orbits "
              f"(total time {t_total:.2f} s).")
    else:
        if t_common is None:
            print(f"⚠ No exact alignment found (error={info['approx_error']:.2e}). "
                  f"Using approximate ratio {info['frac']}.")
            t_common = info['n'] * T_debris
        t_total = t_common

    # Time samples over total simulation
    t_vals = np.linspace(0, t_total, num_points)

    # Unpack debris lambdas
    Xt, Yt, Zt = debrisPos
    Xd = Xt(t_vals)
    Yd = Yt(t_vals) if callable(Yt) else np.full_like(t_vals, Yt)
    Zd = Zt(t_vals)

    # Unpack antenna lambdas
    (X1t, Y1t, Z1t), (X2t, Y2t, Z2t) = antPos
    X1 = X1t(t_vals) if callable(X1t) else np.full_like(t_vals, X1t)
    Y1 = Y1t(t_vals) if callable(Y1t) else np.full_like(t_vals, Y1t)
    Z1 = Z1t(t_vals) if callable(Z1t) else np.full_like(t_vals, Z1t)
    X2 = X2t(t_vals) if callable(X2t) else np.full_like(t_vals, X2t)
    Y2 = Y2t(t_vals) if callable(Y2t) else np.full_like(t_vals, Y2t)
    Z2 = Z2t(t_vals) if callable(Z2t) else np.full_like(t_vals, Z2t)

    # distances to foci
    d1 = np.sqrt((Xd - X1)**2 + (Yd - Y1)**2 + (Zd - Z1)**2)
    d2 = np.sqrt((Xd - X2)**2 + (Yd - Y2)**2 + (Zd - Z2)**2)

    # check inside ellipse
    inside = (d1 + d2) <= 2 * a
    fraction_inside = np.mean(inside)

    return fraction_inside, t_total, info


# -------------------------------
# Plot function (unchanged)
# -------------------------------
def plot_orbits(debrisPos, antPos, T, num_points=500):
    t_vals = np.linspace(0, T, num_points)

    Xt, Yt, Zt = debrisPos
    Xd = Xt(t_vals)
    Yd = Yt(t_vals)
    Zd = Zt(t_vals)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xd, Yd, Zd, label="Debris Orbit", color='orange', linewidth=2)

    for i, (Xf, Yf, Zf) in enumerate(antPos, start=1):
        Xa = Xf(t_vals)
        Ya = Yf if np.isscalar(Yf) else Yf(t_vals)
        Za = Zf(t_vals)
        ax.plot(Xa, Ya, Za, label=f"Antenna {i}", linewidth=1.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Orbital Paths")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    plt.show()


# -------------------------------
# Main function with integration
# -------------------------------
def movingDebris(R, tilt, rotation):
    debrisRange = ant_orbit + R
    cosTilt = np.cos(np.deg2rad(tilt))
    sinTilt = np.sin(np.deg2rad(tilt))
    cosRot = np.cos(np.deg2rad(rotation))
    sinRot = np.sin(np.deg2rad(rotation))

    T = orbitTime(debrisRange)  # period of debris orbit

    Xt = lambda t: debrisRange * (cosRot * np.cos(2*np.pi*t/T) +
                                  sinTilt * sinRot * np.sin(2*np.pi*t/T))
    Yt = lambda t: debrisRange * (sinRot * np.cos(2*np.pi*t/T) +
                                  sinTilt * cosRot * np.sin(2*np.pi*t/T))
    Zt = lambda t: debrisRange * cosTilt * np.sin(2*np.pi*t/T)

    debrisPos = [Xt, Yt, Zt]

    delay = archAngle(R)

    X1t = lambda t: debrisRange * np.sin(2*np.pi*t/ant_orbit_time + delay)
    X2t = lambda t: debrisRange * np.sin(2*np.pi*t/ant_orbit_time - delay)
    Z1t = lambda t: debrisRange * np.cos(2*np.pi*t/ant_orbit_time + delay)
    Z2t = lambda t: debrisRange * np.cos(2*np.pi*t/ant_orbit_time - delay)

    antPos = [[X1t, 0, Z1t], [X2t, 0, Z2t]]

    # --- Calculate fraction inside ellipse ---
    a = np.tan(phi_HPBW/2) * R  # assuming you already know this; replace if needed
    fraction, t_common, info = fraction_inside_ellipse(
        debrisPos, antPos, T, ant_orbit_time, a)

    print(f"Debris is inside ellipse {fraction*100:.6e}% of the time.")
    print(f"Common repeat time ≈ {t_common:.3f} s (ratio ≈ {info['frac']}, error={info['approx_error']:.2e})")

    # --- Plot orbits ---
    plot_orbits(debrisPos, antPos, T)


def main():
    E_TrashSingle = singleImage()
    # ESingleOrbit = singleOrbit()
    # movingDebris(50, 45, 0)



if __name__ == "__main__":
    main()
