import numpy as np
import matplotlib.pyplot as plt
from radiation_pattern import radiation_pattern
from matplotlib.lines import Line2D

def plot_for_report():

    hpbw_el_deg = 27
    hpbw_az_deg = 88

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- TX properties ---
    tx_loc = np.array([0,0,0])

    # --- RX ---
    rx_locs = np.array([[0, -0.03879310344827586, 0], [0, -0.012931034482758622, 0], [0, 0.012931034482758619, 0], [0, 0.03879310344827586, 0]])
    ax.scatter(rx_locs[:, 0], rx_locs[:, 1], rx_locs[:, 2],
               c="b", marker="o", s=50, label="RX")

    # --- TX ---
    ax.scatter(*tx_loc, c="r", marker="^", s=80, label="TX")

    for angle in range(2):
        # --- TX radiation pattern ---
        el_angle, el_gain = radiation_pattern(hpbw_az_deg, include_sidelobes=False)
        az_angle, az_gain = radiation_pattern(hpbw_el_deg, include_sidelobes=True, steering_angle_deg=([-55, 55])[angle])
        az = np.deg2rad(az_angle)
        el = np.deg2rad(el_angle)
        az_pat = az_gain
        el_pat = el_gain

        az_gain = 10 ** ((az_pat - np.max(az_pat)) / 20)
        el_gain = 10 ** ((el_pat - np.max(el_pat)) / 20)

        AZ, EL = np.meshgrid(az, el)
        R = np.outer(el_gain, az_gain)
        X = tx_loc[0] + R * np.cos(EL) * np.cos(AZ)
        Y = tx_loc[1] + R * np.cos(EL) * np.sin(AZ)
        Z = tx_loc[2] + R * np.sin(EL)    
        ax.plot_surface(X, Y, Z, alpha=0.3, color=(["orange", "blue"])[angle], linewidth=0)


    # --- Labels ---
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    
    legend_elem = [Line2D([0], [0], color='orange', lw=4, label='Radiation Pattern (-55) deg'),
                   Line2D([0], [0], color='blue', lw=4, label='Radiation Pattern (55) deg')]
    ax.legend(handles=legend_elem)
    ax.set_title(f"Radar 3D Scene")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-.5, .5)
    ax.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    plt.show()


# plot_for_report()

res = np.arange(-50, 55, 5)
print(res, len(res))