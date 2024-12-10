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
from flask import Flask, request, render_template, send_from_directory

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

# 1. Noise Reduction using Spectral Gating
def spectral_gating(audio, sr, threshold=0.02):
    stft = librosa.stft(audio)
    magnitude, phase = librosa.magphase(stft)
    mask = magnitude > threshold * np.max(magnitude)
    gated_stft = stft * mask
    reduced_audio = librosa.istft(gated_stft)
    return reduced_audio

# 2. Improved Filters using FFT
def apply_fft_filter(audio, sr, cutoff, filter_type):
    fft_audio = fft(audio)
    frequencies = np.fft.fftfreq(len(audio), 1 / sr)
    
    if filter_type == 'low':
        mask = frequencies < cutoff
    elif filter_type == 'high':
        mask = frequencies > cutoff
    elif filter_type == 'band':
        mask = (frequencies > cutoff[0]) & (frequencies < cutoff[1])
    
    fft_audio[~mask] = 0
    filtered_audio = ifft(fft_audio).real
    return filtered_audio

# 3. Vocal Isolation using HPSS
def isolate_vocals(audio, sr):
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive

# Processing and Previewing
# Noise Reduction
reduced_audio = spectral_gating(audio, sr)

# Filters
low_pass_audio = apply_fft_filter(audio, sr, cutoff=300, filter_type='low')
high_pass_audio = apply_fft_filter(audio, sr, cutoff=3000, filter_type='high')
band_pass_audio = apply_fft_filter(audio, sr, cutoff=[500, 3000], filter_type='band')

# Vocal Isolation
harmonic_audio, percussive_audio = isolate_vocals(audio, sr)

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

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        audio, sr = librosa.load(filepath, sr=None)
        
        # Process the audio
        reduced_audio = spectral_gating(audio, sr)
        low_pass_audio = apply_fft_filter(audio, sr, cutoff=300, filter_type='low')
        high_pass_audio = apply_fft_filter(audio, sr, cutoff=3000, filter_type='high')
        band_pass_audio = apply_fft_filter(audio, sr, cutoff=[500, 3000], filter_type='band')
        harmonic_audio, percussive_audio = isolate_vocals(audio, sr)
        
        # Save processed audio
        sf.write(os.path.join(OUTPUT_FOLDER, 'original_audio.wav'), audio, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, 'reduced_audio.wav'), reduced_audio, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, 'low_pass_audio.wav'), low_pass_audio, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, 'high_pass_audio.wav'), high_pass_audio, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, 'band_pass_audio.wav'), band_pass_audio, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, 'harmonic_audio.wav'), harmonic_audio, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, 'percussive_audio.wav'), percussive_audio, sr)
        
        return render_template('index.html', filenames=os.listdir(OUTPUT_FOLDER))

@app.route('/output_audio/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
