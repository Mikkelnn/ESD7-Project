import numpy as np
import matplotlib.pyplot as plt

# ====== Antenna / geometry parameters ======
R = 1.0  # Sphere radius
theta_bw_deg = 27   # Total vertical HPBW (degrees)
phi_bw_deg   = 88   # Total horizontal HPBW (degrees)

# Convert to radians (half-angles)
theta_bw = np.radians(theta_bw_deg)
phi_bw = np.radians(phi_bw_deg)
theta_half = theta_bw / 2
phi_half = phi_bw / 2

# ====== Compute solid angle (steradians) ======
# Elliptical beam approximation: Ω ≈ π/4 * θ_bw * φ_bw
Omega = (np.pi / 4) * theta_bw * phi_bw

# Corresponding spherical volume
V = (R**3 / 3) * Omega

# ====== Generate mesh for beam footprint ======
phi = np.linspace(-phi_half * 1.3, phi_half * 1.3, 300)
theta = np.linspace(0, theta_half * 1.3, 300)
phi, theta = np.meshgrid(phi, theta)

# Elliptical mask for beam region
mask = (theta / theta_half)**2 + (phi / phi_half)**2 <= 1.0

# Sphere coordinates
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)

# Mask points outside beam
x_masked = np.where(mask, x, np.nan)
y_masked = np.where(mask, y, np.nan)
z_masked = np.where(mask, z, np.nan)

# ====== Full sphere for context ======
phi_full = np.linspace(-np.pi, np.pi, 200)
theta_full = np.linspace(0, np.pi, 200)
phi_full, theta_full = np.meshgrid(phi_full, theta_full)

x_sphere = R * np.sin(theta_full) * np.cos(phi_full)
y_sphere = R * np.sin(theta_full) * np.sin(phi_full)
z_sphere = R * np.cos(theta_full)

# ====== Plot ======
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d', proj_type='persp')

# Transparent sphere for context
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.1, linewidth=0)

# Blue beam region
ax.plot_surface(x_masked, y_masked, z_masked, color='royalblue', alpha=0.8, linewidth=0)

# Labels and style
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f"Antenna Field of View (Elliptical Beam)\nθ = {theta_bw_deg}°, φ = {phi_bw_deg}°")

# Add annotation with computed values
text = (
    f"Solid angle Ω = {Omega:.3f} sr\n"
    f"Fraction of full sphere = {Omega / (4*np.pi) * 100:.2f}%\n"
    f"Enclosed volume V = {V:.3f} R³"
)
ax.text2D(0.05, 0.02, text, transform=ax.transAxes, fontsize=10, family="monospace")

# Camera view
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.show()
