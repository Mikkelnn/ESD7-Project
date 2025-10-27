import radarsimpy
from radarsimpy import Radar, Transmitter, Receiver
import radarsimpy.processing as proc
from radarsimpy.simulator import sim_radar

import numpy as np
from scipy import signal, linalg, fft
import plotly.graph_objs as go
from IPython.display import Image
import matplotlib.pyplot as plt
import csv

from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor



# print("`RadarSimPy` used in this example is version: " + str(radarsimpy.__version__))

c = 3e8
f_c = 5.8e9 # center frequency
wavelength = c / f_c

bw = 20.0e6 # bandwidth
t_chirp = 3.2e-6 # chirp time
prp=4.8e-6 # Pulse Repetition Period
pulses = 128

fs = 45e6 # 50e6 # IF fs

r_max = (c * t_chirp) / 2 # calculate the maximum range
delta_R = c / (2 * bw)  # Calculate range resolution (meters / bin)
doppler_max = wavelength / (4 * prp) #((wavelength * (1 / (2 * prp))) / 2)
delta_velocity = wavelength / (2 * pulses * prp)

# print(f"max range: {round(r_max, 2)} m; range resolution: {round(delta_R, 3)} m")
# print(f"max velocity {round(doppler_max, 2)} m/s; velocity resolution: {round(delta_velocity, 3)} m/s")
# print(f"tx time: {prp * pulses}s; sampls/chirp: {round(t_chirp * fs, 2)}")

N_tx = 1
N_rx = 4

tx_channels = []
if (N_tx == 1):
    tx_channels.append(dict(location=(0, 0, 0)))
else:
    for idx in range(0, N_tx):
        tx_channels.append(dict(location=(0, wavelength / 2 * idx - (N_tx - 1) * wavelength / 4, 0)))


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

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=radar.radar_prop["transmitter"].txchannel_prop["locations"][:, 1]
        / wavelength,
        y=radar.radar_prop["transmitter"].txchannel_prop["locations"][:, 2]
        / wavelength,
        mode="markers",
        name="Transmitter",
        opacity=0.7,
        marker=dict(size=10),
    )
)

fig.add_trace(
    go.Scatter(
        x=radar.radar_prop["receiver"].rxchannel_prop["locations"][:, 1] / wavelength,
        y=radar.radar_prop["receiver"].rxchannel_prop["locations"][:, 2] / wavelength,
        mode="markers",
        opacity=1,
        name="Receiver",
    )
)

fig.update_layout(
    title="Array configuration",
    xaxis=dict(title="y (λ)"),
    yaxis=dict(title="z (λ)", scaleanchor="x", scaleratio=1),
)

rcs = 10

def runSimulation(params):
    #Do a lot of simulations
    targets = []
    for i, (range, velocity, angle) in enumerate(params):
        cos_val = range * np.cos(np.radians(angle))
        sin_val = range * np.sin(np.radians(angle))

        targetLoop = dict(
            location=(
                sin_val,
                cos_val,
                0,
            ),
            speed=(velocity, 0, 0),
            rcs=rcs,
            phase=0,
        )

        targets.append(targetLoop)

    data = sim_radar(radar, targets)
    timestamp = data["timestamp"]
    baseband = data["baseband"] #+ data["noise"]
    return baseband
