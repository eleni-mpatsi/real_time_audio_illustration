import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import librosa

# Initialize PyAudio
p = pyaudio.PyAudio()

# Set the microphone device index
mic_device_index = 1

# Define constants
WINDOW_SIZE = 2048  
CHANNELS = 1  
RATE = 44100  

# Constants for spectrogram
FFT_FRAMES_IN_SPEC = 20  
HOP_LENGTH = WINDOW_SIZE // FFT_FRAMES_IN_SPEC 

# Constants for plot
MFCC_FRAMES_IN_PLOT = 13  
MFCC_HOP_LENGTH = WINDOW_SIZE // MFCC_FRAMES_IN_PLOT  

# global variables
global_blocks = np.zeros((FFT_FRAMES_IN_SPEC, WINDOW_SIZE))  # for storing audio frames for spectrogram
mfcc_frames = np.zeros((MFCC_FRAMES_IN_PLOT, 13))  # for storing MFCC frames for plot
win = np.hamming(WINDOW_SIZE)  # Window function for FFT analysis
spec_img = np.zeros((WINDOW_SIZE // 2, FFT_FRAMES_IN_SPEC))  # Array to store spectrogram
user_terminated = False  # Flag to indicate if the user has terminated the program

#------------------------------------------------------------------------------------

def callback(in_data, frame_count, time_info, status):
    global global_blocks, mfcc_frames, win, spec_img
    numpy_block_from_bytes = np.frombuffer(in_data, dtype='int16')
    
    block_for_speakers = np.zeros((numpy_block_from_bytes.size, CHANNELS), dtype='int16')
    block_for_speakers[:, 0] = numpy_block_from_bytes
    
    # Perform FFT and compute spectrogram
    if len(win) == len(numpy_block_from_bytes):
        frame_fft = np.fft.fft(win * numpy_block_from_bytes)
        p = np.abs(frame_fft) * 2 / np.sum(win)
        
        # Translate to dB
        fft_frame = 20 * np.log10(p[:WINDOW_SIZE // 2] / 32678)
        
        # Update spectrogram
        spec_img = np.roll(spec_img, -1, axis=1)
        spec_img[:, -1] = fft_frame[::-1]
        
        # Update global blocks
        global_blocks = np.roll(global_blocks, -1, axis=0)
        global_blocks[-1, :] = numpy_block_from_bytes
        
        # Compute MFCC
        if global_blocks.shape[0] == FFT_FRAMES_IN_SPEC:
            s = np.reshape(global_blocks, WINDOW_SIZE * FFT_FRAMES_IN_SPEC)
            mfcc_block = librosa.feature.mfcc(y=s, sr=RATE, n_mfcc=13, n_fft=WINDOW_SIZE, hop_length=MFCC_HOP_LENGTH)
            mfcc_frames = np.roll(mfcc_frames, -1, axis=0)
            mfcc_frames[-1, :] = mfcc_block[:, -1]
    
    return (block_for_speakers, pyaudio.paContinue)


def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    print('pressed:', k)
    
    if k == 's' or k == 'S':
        global user_terminated
        user_terminated = True
        print('user_terminated 1:', user_terminated )

output = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    output=False,
    input=True,
    input_device_index=mic_device_index,
    frames_per_buffer=WINDOW_SIZE,
    stream_callback=callback,
    start=False
)

output.start_stream()

threaded_input = Thread(target=user_input_function)
threaded_input.start()

# plot spectrogram and MFCC's
while output.is_active() and not user_terminated:
    # Create a figure with two subplots
    fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
    
    # Plot the spectrogram
    ax[0].imshow(spec_img[WINDOW_SIZE // 4:, :], aspect='auto')
    ax[0].set_title('Spectrogram')
    ax[0].set_xticks([])
    
    # Plot the centroid line on the spectrogram
    s = np.reshape(global_blocks, WINDOW_SIZE * FFT_FRAMES_IN_SPEC)
    c = librosa.feature.spectral_centroid(y=s, sr=RATE, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE)
    centroid_values = c[0][-20:] / 100
    ax[0].plot(centroid_values, color='r', linewidth=2)
    ax[0].set_title('Spectrogram with Centroid Line')
    
    # Plot the MFCC coefficients
    mfcc_img = ax[1].imshow(mfcc_frames.T, aspect='auto', cmap='viridis', origin='lower', extent=[0, len(mfcc_frames), 0, mfcc_frames.shape[0]])
    ax[1].set_title('MFCC Coefficients')
    fig.colorbar(mfcc_img, ax=ax[1])
    
    plt.show()
    plt.pause(0.01)

print('stopping audio')
output.stop_stream()
