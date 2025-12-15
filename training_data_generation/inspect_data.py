import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import fft
# import matplotlib
# matplotlib.use('gtk3agg')

# from dsp_mini import range_doppler_fft

ROOT_PATH = Path("/home/mikkel/repoes/zero_one/training_data/")

# label_files = Path(ROOT_PATH / "labels").glob("*")

# zero_count = 0
# nonzero_count = 0
# for f in label_files:
#     label = np.load(f)
#     if np.sum(label) != 0:
#         nonzero_count += 1
#     else:
#         zero_count += 1

# print(f"zero labels: {zero_count}, non-zero labels: {nonzero_count}")
# exit()
# 394d2bf4-8988-4b4e-9663-845550c9cfb7.npy  7cc0543a-b305-4265-a843-3c06a0b81641.npy  bce51239-c4fc-482f-9413-9b398d9fffaa.npy  fd1c1cdc-0583-4b5a-9abf-5e0b3323d313.npy
file_to_inspect = "394d2bf4-8988-4b4e-9663-845550c9cfb7.npy"

input_path = ROOT_PATH / "input" /  file_to_inspect
label_path = ROOT_PATH / "labels" / file_to_inspect

# input_path = "test.npy"

input_sweep = np.load(input_path)
label = np.load(label_path)

print(f"input shape: {input_sweep.shape}")

label_scaled = [int(label[0] * 1000), int(label[1] * 7500)]
print(f"raw: {label}, scaled: {label_scaled}")

# exit()

# input = np.sum(input, axis=0)
# input = np.abs(input)
# input_power = ((input**2)/50) #p=u²/r
# input_dbm = 10 * np.log10(input_power) + 30

# input = range_doppler_fft(input)

doppler_bins = input_sweep[0].shape[0]
range_bins = input_sweep[0].shape[1]


c = 3e8
f_c = 5.8e9 # center frequency

bw = 20.0e6 # bandwidth
t_chirp = 3.2e-6 # chirp time
prp=4.8e-6
fs = 45e6 # 50e6 # IF fs

max_range = (c * fs * t_chirp / bw / 2)
range_axis = np.linspace(0, max_range, range_bins, endpoint=False)

unambiguous_speed = (c / (prp * f_c * 4))
doppler_axis = np.linspace(-unambiguous_speed, unambiguous_speed, doppler_bins)

print(f"max vel: {unambiguous_speed}, {(unambiguous_speed*2)/doppler_bins} / bin; max range: {max_range}, {max_range / range_bins} / bin")

for idx, input in enumerate(input_sweep):
    # input = input_sweep[0]
    input = fft.fftshift(input, axes=0)
    input = 20 * np.log10(input)

    plt.figure(idx)
    plt.imshow(input, cmap='viridis', aspect='auto', extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]])
    # plt.imshow(input_dbm, cmap='viridis', aspect='auto')
    plt.colorbar(label='Amplitude (dBm)')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Range-Doppler Map (sum over angle)')

plt.show()
