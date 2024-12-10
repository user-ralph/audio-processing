# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import librosa
import librosa.display
import soundfile as sf
from scipy.fft import fft, ifft
import os

# Function to preview audio (Note: ipd.Audio is not available outside Colab/Jupyter)
def preview_audio(audio, sr, filename):
    output_dir = 'output_audio'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, audio, sr)
    print(f"Audio saved as {filepath}")

# Load an audio file
file_path = 'samples/AudioTrimmedB.mp3'  # Update this with the correct file path
audio, sr = librosa.load(file_path, sr=None)  # Load audio with its original sampling rate

# 1. Noise Reduction
def noise_reduction(audio, sr, noise_factor=0.02):
    noise = np.random.normal(0, noise_factor, audio.shape)
    reduced_audio = audio - noise
    return reduced_audio

# 2. Filters
def butter_filter(cutoff, sr, filter_type, order=5):
    nyquist = 0.5 * sr
    if isinstance(cutoff, list):
        normal_cutoff = [c / nyquist for c in cutoff]
    else:
        normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return b, a

def apply_filter(audio, sr, cutoff, filter_type):
    b, a = butter_filter(cutoff, sr, filter_type)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

# 3. Vocal Isolation using HPSS
def isolate_vocals(audio, sr):
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive

# 4. Frequency Domain Manipulation
def frequency_domain_manipulation(audio, sr, threshold=0.01):
    fft_audio = fft(audio)
    magnitude = np.abs(fft_audio)
    magnitude[magnitude < threshold * np.max(magnitude)] = 0
    processed_audio = ifft(fft_audio).real
    return processed_audio

# 5. Extract Tonal Features
def extract_tonal_features(audio, sr, freq_range=(1000, 2000)):
    fft_audio = fft(audio)
    frequencies = np.fft.fftfreq(len(audio), 1 / sr)
    mask = (frequencies > freq_range[0]) & (frequencies < freq_range[1])
    fft_filtered = fft_audio * mask
    tonal_features_audio = ifft(fft_filtered).real
    return tonal_features_audio, frequencies[mask]

# Processing and Previewing
# Noise Reduction
reduced_audio = noise_reduction(audio, sr)

# Filters
low_pass_audio = apply_filter(audio, sr, cutoff=300, filter_type='low')
high_pass_audio = apply_filter(audio, sr, cutoff=3000, filter_type='high')
band_pass_audio = apply_filter(audio, sr, cutoff=[500, 3000], filter_type='band')

# Vocal Isolation
harmonic_audio, percussive_audio = isolate_vocals(audio, sr)

# Frequency Manipulation
background_removed_audio = frequency_domain_manipulation(audio, sr)

# Tonal Features
tonal_audio, tonal_frequencies = extract_tonal_features(audio, sr)

# Save and Preview Processed Audio
print("Preview Original Audio")
preview_audio(audio, sr, 'original_audio.wav')

print("Preview Noise-Reduced Audio")
preview_audio(reduced_audio, sr, 'reduced_audio.wav')

print("Preview Low-Pass Filtered Audio")
preview_audio(low_pass_audio, sr, 'low_pass_audio.wav')

print("Preview High-Pass Filtered Audio")
preview_audio(high_pass_audio, sr, 'high_pass_audio.wav')

print("Preview Band-Pass Filtered Audio")
preview_audio(band_pass_audio, sr, 'band_pass_audio.wav')

print("Preview Harmonic (Vocals) Audio")
preview_audio(harmonic_audio, sr, 'harmonic_audio.wav')

print("Preview Percussive (Background Noise) Audio")
preview_audio(percussive_audio, sr, 'percussive_audio.wav')

print("Preview Background Noise Removed Audio")
preview_audio(background_removed_audio, sr, 'background_removed_audio.wav')

print("Preview Tonal Features Audio")
preview_audio(tonal_audio, sr, 'tonal_audio.wav')

def plot_waveform(audio, sr, title):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

# Plot original and processed waveforms
plot_waveform(audio, sr, "Original Audio")
plot_waveform(reduced_audio, sr, "Noise-Reduced Audio")
plot_waveform(low_pass_audio, sr, "Low-Pass Filtered Audio")
plot_waveform(high_pass_audio, sr, "High-Pass Filtered Audio")
plot_waveform(band_pass_audio, sr, "Band-Pass Filtered Audio")
plot_waveform(harmonic_audio, sr, "Harmonic (Vocals) Audio")
plot_waveform(percussive_audio, sr, "Percussive (Background Noise) Audio")
plot_waveform(background_removed_audio, sr, "Background Noise Removed Audio")
plot_waveform(tonal_audio, sr, "Tonal Features Audio")
