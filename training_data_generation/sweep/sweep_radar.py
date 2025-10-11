from radiation_pattern import radiation_pattern
from plot_3d_scene import  plot_radar_scene

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from radarsimpy import Radar, Transmitter, Receiver
# from radarsimpy.simulator import sim_radar

c = 3e8
f_c = 5.8e9 # center frequency
wavelength = c / f_c

bw = 20.0e6 # bandwidth
t_chirp = 3.2e-6 # chirp time
prp=4.8e-6 # Pulse Repetition Period
pulses = 128

fs = 45e6 # 50e6 # IF fs

hpbw_el_deg = 27
hpbw_az_deg = 88

r_max = (c * t_chirp) / 2 # calculate the maximum range
delta_R = c / (2 * bw)  # Calculate range resolution (meters / bin)
doppler_max = wavelength / (4 * prp) #((wavelength * (1 / (2 * prp))) / 2)
delta_velocity = wavelength / (2 * pulses * prp)

# print(f"max range: {round(r_max, 2)} m; range resolution: {round(delta_R, 3)} m")
# print(f"max velocity {round(doppler_max, 2)} m/s; velocity resolution: {round(delta_velocity, 3)} m/s")
# print(f"tx time: {prp * pulses}s; sampls/chirp: {round(t_chirp * fs, 2)}")


el_angle, el_gain = radiation_pattern(hpbw_el_deg, include_sidelobes=True, steering_angle_deg=0)
az_angle, az_gain = radiation_pattern(hpbw_az_deg, include_sidelobes=False)

N_tx = 1
N_rx = 4

tx_channels = []
if (N_tx == 1):
    tx_channels.append(dict(
        location=(0, 0, 0), 
        azimuth_angle=az_angle,
        azimuth_pattern=az_gain,
        elevation_angle=el_angle,
        elevation_pattern=el_gain,
    ))
else:
    for idx in range(0, N_tx):
        tx_channels.append(dict(
            location=(0, wavelength / 2 * idx - (N_tx - 1) * wavelength / 4, 0),
            azimuth_angle=az_angle,
            azimuth_pattern=az_gain,
            elevation_angle=el_angle,
            elevation_pattern=el_gain,
        ))


rx_channels = []
for idx in range(0, N_rx):
    rx_channels.append(
        dict(
            location=(0, wavelength / 2 * idx - (N_rx - 1) * wavelength / 4, 0),
        )
    )

tx = Transmitter(
    f=[f_c - (bw/2), f_c + (bw/2)],
    t=[0, t_chirp],
    tx_power=40, # 40
    prp=prp,
    pulses=pulses,
    channels=tx_channels
)

rx = Receiver(
    fs=fs,
    noise_figure=0, # 8
    rf_gain=20,
    load_resistor=50,
    baseband_gain=30,
    channels=rx_channels
)

radar = Radar(transmitter=tx, receiver=rx)

target_list = [
    dict(
        location=(10, 0, 0),
        speed=(5, 0, 0),
        rcs=10,
        phase=0,
    )
]

plot_radar_scene(radar, target_list, show_pattern=False, pulse_idx=None)
