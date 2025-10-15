# Script calculating parameters for the stochastic analysis of how much trash we expect to detect in LEO
# Debris is assumed to be uniformly distributed throughout LEO
# ALL UNITS ARE SI!!!!!!!

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, floor
from collections import deque

# ----------------------- Global parameters (user-set) -----------------------
r_Earth = 6371e3
r_lower = 180e3 + r_Earth
r_upper = 2000e3 + r_Earth
V_LEO = (4.0/3.0) * np.pi * (np.power(r_upper, 3) - np.power(r_lower, 3))

trash_LEO = 934000          # number of debris pieces in LEO
density = trash_LEO / V_LEO # debris per m^3

# Radar / beam parameters
ant_range = 1000.0                  # meters (you confirmed 1000 m)
theta_HPBW = np.deg2rad(27.0)       # full HPBW (radians)
phi_HPBW   = np.deg2rad(88.0)       # full HPBW (radians)

# Voxelization / sampling parameters
voxel_size = 10.0         # m (set to 1.0 for high precision, but cost increases)
orbit_samples = 720       # number of satellite positions along orbit
cone_radial_samples = 40  # subdivisions along cone radius (0..R)
cone_angular_samples = 80 # subdivisions across each angular axis (per side)
beam_tilt_deg = 0.0       # keep 0 for zenith pointing

# Orbit
orbit_altitude = 550e3
orbit_radius_m = r_Earth + orbit_altitude

# Derived
half_theta = theta_HPBW / 2.0
half_phi   = phi_HPBW / 2.0
beam_tilt_rad = np.deg2rad(beam_tilt_deg)
angles = np.linspace(0.0, 2.0*pi, orbit_samples, endpoint=False)
# ---------------------------------------------------------------------------


# ----------------------------- helper functions -----------------------------
def fibonacci_sphere(n):
    """Return n approximately uniform directions on the unit sphere (shape (n,3))."""
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i + 0.5) / n)
    theta = 2.0 * pi * ((1.0 + 5.0**0.5) / 2.0 * i) % (2.0 * pi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T


def compute_beam_solid_angle_by_sampling(half_theta, half_phi, n_dirs=60000):
    """
    Estimate the rectangular beam solid angle by sampling directions on the unit sphere and
    testing whether each direction lies within the beam centered on boresight=(0,0,1).
    """
    dirs = fibonacci_sphere(n_dirs)
    # local boresight axis
    axis = np.array([0.0, 0.0, 1.0])
    # build local orthonormal frame (e1,e2) used to measure angular offsets
    def orthonormal_frame(axis_):
        if abs(axis_[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, axis_)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(axis_, e1)
        return e1, e2
    e1, e2 = orthonormal_frame(axis)
    adot = dirs.dot(axis)
    mask_forward = adot > 0
    idxs = np.nonzero(mask_forward)[0]
    covered = np.zeros(dirs.shape[0], dtype=bool)
    if idxs.size > 0:
        p = dirs[idxs] - adot[idxs, None] * axis[None, :]
        x = p.dot(e1)
        y = p.dot(e2)
        ang_x = np.arctan2(x, adot[idxs])
        ang_y = np.arctan2(y, adot[idxs])
        inside = (np.abs(ang_x) <= half_theta) & (np.abs(ang_y) <= half_phi)
        covered[idxs[inside]] = True
    frac = covered.mean()
    omega = 4.0 * np.pi * frac
    return omega, frac


def rotation_matrix(axis, angle):
    """Return 3x3 rotation matrix rotating around 'axis' by 'angle' (radians)."""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle); s = np.sin(angle)
    R = np.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])
    return R


def sample_points_in_rectangular_cone(R, half_theta, half_phi, n_rad, n_theta, n_phi):
    """
    Sample points inside a rectangular-angular cone (boresight along +x in the local frame).
    Returns an (N,3) array of points in local cone coordinates where boresight==+x.
    """
    r_vals = np.linspace(0.0, R, n_rad)
    theta_vals = np.linspace(-half_theta, half_theta, n_theta)
    phi_vals = np.linspace(-half_phi, half_phi, n_phi)
    pts = []
    for r in r_vals:
        if r == 0.0:
            pts.append((0.0, 0.0, 0.0))
            continue
        # tangent-plane mapping to directions
        for a in theta_vals:
            for b in phi_vals:
                dx = 1.0
                dy = np.tan(a)
                dz = np.tan(b)
                vec = np.array([dx, dy, dz])
                vec = vec / np.linalg.norm(vec)
                pts.append(tuple(vec * r))
    return np.array(pts, dtype=float)


def local_frames_for_orbit_angle(ang):
    """
    Return satellite position and a 3x3 rotation matrix M that transforms local-cone
    coordinates (boresight along +x, lateral axes y,z) into world coords:
      world_point = sat_pos + M @ local_point
    Boresight here is selected to be local zenith (pointing away from Earth center).
    """
    sat = np.array([orbit_radius_m * cos(ang), orbit_radius_m * sin(ang), 0.0])
    zenith = sat / np.linalg.norm(sat)          # local zenith (outward)
    tangent = np.array([-sin(ang), cos(ang), 0.0])
    tangent = tangent / np.linalg.norm(tangent)
    binormal = np.cross(zenith, tangent)
    binormal = binormal / np.linalg.norm(binormal)
    boresight_vec = zenith
    y_local = binormal
    z_local = np.cross(boresight_vec, y_local)
    M = np.column_stack((boresight_vec, y_local, z_local))  # 3x3
    return sat, M
# ---------------------------------------------------------------------------


# ----------------------------- analysis functions ---------------------------
def singleImage():
    """Estimate expected debris in a single still image (using sampled beam solid angle)."""
    omega_single, frac = compute_beam_solid_angle_by_sampling(half_theta, half_phi, n_dirs=60000)
    V_ant = (omega_single / 3.0) * (ant_range ** 3)   # correct truncated-cone volume
    E_trashImage = density * V_ant
    print("=== Single image estimate ===")
    print(f"Trash density: {density:.3e} pieces/m^3")
    print(f"Beam solid angle (sr): {omega_single:.6f} (covered fraction of sphere = {frac:.6f})")
    print(f"Antenna truncated-cone volume (m^3): {V_ant:.6e}")
    print(f"Expected trash pieces in a single measurement: {E_trashImage:.6e}")
    print("")


def singleOrbit():
    """
    Compute voxel-union for sweeping the cone for one full orbit.
    Returns the voxelized swept volume (m^3) and number of occupied voxels.
    """
    print("=== Single orbit voxel union (this may take some time depending on voxel_size) ===")
    # sample point cloud for a single cone in local frame
    cone_local = sample_points_in_rectangular_cone(
        ant_range,
        half_theta,
        half_phi,
        cone_radial_samples,
        cone_angular_samples // 2,
        cone_angular_samples // 2
    )
    # apply tilt if requested (beam_tilt_rad)
    if beam_tilt_rad != 0.0:
        tilt_rotation = rotation_matrix(np.array([0.0, 1.0, 0.0]), beam_tilt_rad)
        cone_local = (cone_local @ tilt_rotation.T)

    occupied = set()  # sparse container for voxel indices

    # iterate satellite positions
    for ang in angles:
        sat_pos, M = local_frames_for_orbit_angle(ang)
        # transform cone points to world coords
        pts_world = sat_pos[None, :] + cone_local @ M.T  # shape (N_pts, 3)
        # convert to voxel indices (floor division)
        idxs = np.floor(pts_world / voxel_size).astype(int)
        # add voxel indices to set
        # vectorized unique per satellite would be faster, but keep simple and safe:
        for ix, iy, iz in idxs:
            occupied.add((int(ix), int(iy), int(iz)))

    n_vox = len(occupied)
    voxel_volume = voxel_size ** 3
    V_swept_m3 = n_vox * voxel_volume

    # expected debris seen in one orbit ≈ density * V_swept_m3
    E_trashOrbit = density * V_swept_m3

    print(f"Voxel size: {voxel_size} m")
    print(f"Orbit samples: {orbit_samples}")
    print(f"Number of occupied voxels: {n_vox:,}")
    print(f"Voxelized swept volume: {V_swept_m3:.6e} m^3 = {V_swept_m3/1e9:.6e} km^3")
    print(f"Expected trash pieces observed during one orbit (assuming stationary debris): {E_trashOrbit:.6e}")
    print("")
    return V_swept_m3, n_vox, E_trashOrbit


def main():
    singleImage()
    V_swept_m3, n_vox, E_trashOrbit = singleOrbit()


if __name__ == "__main__":
    main()
