import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import io
# from mpl_toolkits.mplot3d import Axes3D

import matplotlib
# matplotlib.use('Agg')

def plot_radar_scene(radar, targets, show_pattern=False, pulse_idx=None, t_offset=0):
    """
    Plot a 3D radar scene for a RadarSimPy radar object.

    Parameters
    ----------
    radar : radarsimpy.Radar
        Configured radar object.
    targets : list of dict
        Each dict must contain:
            {
                'location': np.array([x, y, z]),   # [m]
                'velocity': np.array([vx, vy, vz]) # [m/s]
            }
    show_pattern : bool, optional
        If True, draw TX radiation pattern (default: False).
    pulse_idx : int or None, optional
        If int  -> display snapshot for that pulse.
        If None -> display for the center pulse.
    t_offset : adds offset to time displayed in title in seconds
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- TX properties ---
    tx = radar.radar_prop["transmitter"].txchannel_prop
    tx_loc = np.array(tx["locations"][0])

    # --- Waveform timing info ---
    wf = radar.radar_prop["transmitter"].waveform_prop
    n_pulses = wf["pulses"]
    pulse_start = wf["pulse_start_time"]
    prp = wf["prp"]

    # Determine pulse index and time
    if pulse_idx is not None:
        pulse_idx = int(np.clip(pulse_idx, 0, n_pulses - 1))
    else:
        pulse_idx = n_pulses // 2  # center pulse
    t_sel = pulse_start[pulse_idx] + 0.5 * prp[pulse_idx]

    # --- RX ---
    rx = radar.radar_prop["receiver"].rxchannel_prop
    rx_locs = np.array(rx["locations"])
    ax.scatter(rx_locs[:, 0], rx_locs[:, 1], rx_locs[:, 2],
               c="b", marker="o", s=50, label="RX")

    # --- TX ---
    ax.scatter(*tx_loc, c="r", marker="^", s=80, label="TX")

    # --- TX radiation pattern ---
    if show_pattern:
        az = np.deg2rad(tx["az_angles"][0])
        el = np.deg2rad(tx["el_angles"][0])
        az_pat = tx["az_patterns"][0]
        el_pat = tx["el_patterns"][0]

        az_gain = 10 ** ((az_pat - np.max(az_pat)) / 20)
        el_gain = 10 ** ((el_pat - np.max(el_pat)) / 20)

        AZ, EL = np.meshgrid(az, el)
        R = np.outer(el_gain, az_gain)
        X = tx_loc[0] + R * np.cos(EL) * np.cos(AZ)
        Y = tx_loc[1] + R * np.cos(EL) * np.sin(AZ)
        Z = tx_loc[2] + R * np.sin(EL)
        ax.plot_surface(X, Y, Z, alpha=0.3, color="orange", linewidth=0)

    # --- Targets ---
    for i, tgt in enumerate(targets):
        p0 = np.array(tgt["location"])
        v = np.array(tgt["speed"])
        p = p0 + v * t_sel  # position at current pulse
        ax.scatter(*p, marker="x", s=60, label=f"Target {i}")
        # Draw a short line in the target’s velocity direction (forward only)
        dir_scale = 1  # [s] scale factor for visualization, adjust for clarity
        p_dir = p + (v /  np.linalg.norm(v)) * dir_scale  # extend only forward in velocity direction
        ax.plot([p[0], p_dir[0]], [p[1], p_dir[1]], [p[2], p_dir[2]], "k--", alpha=0.6)

    # --- Labels ---
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_title(f"Radar 3D Scene @ Pulse {pulse_idx} (t={(t_offset + t_sel)*1e6:.2f} µs)")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_zlim(-11, 11)

    fig.canvas.draw()
    buf = io.BytesIO()
    fig.canvas.print_rgba(buf)
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[..., :3]  # drop alpha if you want RGB
    plt.close(fig)

    return image

def save_frames_mp4(frames, path=None):
    print("Saving frames...")

    # Save frames as a video
    metadata = dict(title='RADAR 3D Animation')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    fig = plt.figure()
    plt.axis('off')  # hide axes for video
    if path is None:
        path = "3d_movie.mp4"

    with writer.saving(fig, path, dpi=150):
        for frame in frames:
            plt.imshow(frame)
            writer.grab_frame()

def show_all_figs(frames):
    for i, frame in enumerate(frames):
        plt.figure(i)
        plt.imshow(frame)
    plt.show()