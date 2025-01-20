import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from encoder import RPE_frame_st_coder
from decoder import RPE_frame_st_decoder


# Read the .wav file
sample_rate, data = wavfile.read("./material/ena_dio_tria.wav")

# Ensure data is mono (if stereo, take one channel)
if len(data.shape) > 1:  # Stereo
    data = data[:, 0]

# Define the frame size
frame_size = 160

# Process data in frames
frames = []
for i in range(0, len(data), frame_size):
    frame = data[i : i + frame_size]
    frames.append(frame)

# Convert frames to a numpy array for easier manipulation
frames = np.array(frames, dtype=object)

# Example: Print number of frames
print(f"Number of frames: {len(frames)}", frames[0].shape)

s0 = np.array(frames, dtype=object)
d = np.zeros(160)
for frame in range(len(frames) - 1):
    LARc, d = RPE_frame_st_coder(frames[frame], d)
    s0[frame] = RPE_frame_st_decoder(LARc, d)
    #print("First sample for frame " + str(frame) + " is " + str(frames[frame][1]))

decoded_signal = np.array(s0)
decoded_signal = np.concatenate(decoded_signal)
print(decoded_signal.shape)

# Sampling rate of the signal (e.g., 8000 Hz for 8 kHz audio)
sampling_rate = 8000

# Normalize the signal to fit within the range of 16-bit audio
max_amplitude = np.iinfo(np.int16).max
decoded_signal = (decoded_signal / np.max(np.abs(decoded_signal))) * max_amplitude

# Convert to int16 format
decoded_signal = decoded_signal.astype(np.int16)

# Write to a WAV file
filename = "decoded_signal.wav"
wavfile.write(filename, sampling_rate, decoded_signal)

print("Decoded signal has been written to", filename)

# plot original vs. decoded signal
plt.subplot(2, 1, 1)
plt.plot(data[600:2000], label="Original Signal")
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.plot(decoded_signal[600:2000], label="Decoded Signal")
plt.title('Decoded Signal')

plt.tight_layout()
plt.show()