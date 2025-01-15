import numpy as np
from scipy.io import wavfile
from encoder import RPE_frame_slt_coder
from decoder import RPE_frame_slt_decoder

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
