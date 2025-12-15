import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
# from scipy import fft

# from dsp_mini import range_doppler_fft

ROOT_PATH = Path("/home/mikkel/repoes/zero_one/validate_data/")

label_files = Path(ROOT_PATH / "labels").glob("*")

zero_count = 0
nonzero_count = 0
for f in label_files:
    label = np.load(f)
    if np.sum(label) != 0:
        nonzero_count += 1
    else:
        zero_count += 1

print(f"zero labels: {zero_count}, non-zero labels: {nonzero_count}")
exit()

file_to_inspect = "9376d46d-f701-4a4b-a8ac-4a8260c416e4.npy"

# input_path = ROOT_PATH + "input/" +  file_to_inspect
# label_path = ROOT_PATH + "labels/" +  file_to_inspect

input_path = "test.npy"

input = np.load(input_path)
# label = np.load(label_path)

# label_scaled = [int(label[0] * 800), int(label[1] * 7500)]
# print(f"raw: {label}, scaled: {label_scaled}")


input = input[0]

print(f"shape: {input.shape}")

# input = np.sum(input, axis=0)
# input = np.abs(input)
# input_power = ((input**2)/50) #p=u²/r
# input_dbm = 10 * np.log10(input_power) + 30

input = range_doppler_fft(input)
input = fft.fftshift(input, axes=0)
input = 20 * np.log10(input)

doppler_bins = input.shape[0]
range_bins = input.shape[1]


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

plt.figure(1)
plt.imshow(input, cmap='viridis', aspect='auto', extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]])
# plt.imshow(input_dbm, cmap='viridis', aspect='auto')
plt.colorbar(label='Amplitude (dBm)')
plt.xlabel('Range (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Range-Doppler Map (sum over angle)')

plt.show()