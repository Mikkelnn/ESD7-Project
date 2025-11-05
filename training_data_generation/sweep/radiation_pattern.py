import numpy as np
import matplotlib.pyplot as plt

def radiation_pattern(hpbw_deg, include_sidelobes=False, steering_angle_deg=0, theta_range=(-90, 90)):
    theta = np.linspace(theta_range[0], theta_range[1], 181)
    # theta_rad = np.radians(theta)
    theta_rad = np.radians(theta - steering_angle_deg)  # Apply steering offset
    
    if include_sidelobes:
        # Approximate aperture pattern using sinc
        # HPBW ≈ 50.8 / D (in deg), so k = π / θ_null
        theta_null = np.radians(hpbw_deg) / 0.885  # first null spacing approx
        k = np.pi / theta_null
        gain = (np.sinc(k * theta_rad / np.pi))**2
    else:
        # Gaussian mainlobe only
        sigma = hpbw_deg / (2 * np.sqrt(2 * np.log(2)))
        gain = np.exp(-0.5 * (theta / sigma)**2)

    gain /= np.max(gain)
    gain_db = 20 * np.log10(np.clip(gain, 1e-6, 1)) + 10.97 # avoid log(0)
    return theta, gain_db


def plot_cartesian(ax, theta, gain_db, title, xlabel, hpbw_lines=True):
    ax.plot(theta, gain_db)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Gain (dB)')
    ax.set_ylim(-40, 15)
    ax.grid(True)

    if hpbw_lines:
        # Determine -3 dB level relative to peak
        peak = np.max(gain_db)
        level = peak - 3

        # Horizontal line at -3 dB relative level
        ax.axhline(level, color='r', linestyle='--', linewidth=1)

        # Find intersection points
        diff = gain_db - level
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        intersections = []
        for i in sign_changes:
            x0, x1 = theta[i], theta[i + 1]
            y0, y1 = gain_db[i], gain_db[i + 1]
            x_cross = x0 + (x1 - x0) * (level - y0) / (y1 - y0)
            intersections.append(x_cross)

        # Vertical lines at intersections
        for x in intersections:
            ax.axvline(x, color='r', linestyle='--', linewidth=1)

        # Annotate HPBW if two crossings exist
        if len(intersections) >= 2:
            hpbw = abs(intersections[-1] - intersections[0])
            ax.text(0.05, 0.9, f'HPBW ≈ {hpbw:.2f}°', transform=ax.transAxes, color='r')
        

def plot_polar(ax, theta, gain_db, title):
    # Convert -90–90° to 0–180° span for plotting
    theta_rad = np.radians(theta)
    gain_lin = 10 ** (gain_db / 20)
    ax.plot(theta_rad, gain_lin)
    ax.set_title(title, va='bottom')
    ax.set_theta_zero_location('N')
    # ax.set_theta_direction(-1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.grid(True)

def plot_pattern(hpbw_az_deg, hpbw_el_deg):
    theta_el, gain_el = radiation_pattern(hpbw_el_deg, include_sidelobes=True, steering_angle_deg=0)
    theta_az, gain_az = radiation_pattern(hpbw_az_deg, include_sidelobes=False)

    fig = plt.figure(figsize=(10, 8))

    # Azimuth
    ax1 = fig.add_subplot(2, 2, 1)
    plot_cartesian(ax1, theta_az, gain_az, f'Azimuth Pattern (HPBW={hpbw_az_deg}°)', 'Azimuth Angle (°)')

    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    plot_polar(ax2, theta_az, gain_az, 'Azimuth Polar')

    # Elevation
    ax3 = fig.add_subplot(2, 2, 3)
    plot_cartesian(ax3, theta_el, gain_el, f'Elevation Pattern (HPBW={hpbw_el_deg}°)', 'Elevation Angle (°)', hpbw_lines=True)
    
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    plot_polar(ax4, theta_el, gain_el, 'Elevation Polar')


    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    plot_pattern(hpbw_az_deg=88, hpbw_el_deg=27)
