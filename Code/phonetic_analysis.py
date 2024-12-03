import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Set the speaker directory and mic type
speaker_dir = '/media/saketh/New Volume/SAL/Dataset/M01/Session1'
mic_type = 'headMic'  # Change to 'headMic' for wav_headMic/phn_headMic

# Define directory names based on mic type
wav_dir = f'wav_{mic_type}'
phn_dir = f'phn_{mic_type}'

# Load alignment information
alignment_file = os.path.join(speaker_dir, 'alignment.txt')
alignment_data = pd.read_csv(alignment_file, sep=' ', header=None)
alignment_data.columns = ['array_dir', 'offset']

# Load audio files
audio_files = []
dir_path = os.path.join(speaker_dir, wav_dir)
audio_files.extend([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')])

# Load phonetic transcriptions
phn_data = {}
dir_path = os.path.join(speaker_dir, phn_dir)
phn_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.PHN') or f.endswith('.phn')]
for phn_file in phn_files:
    with open(phn_file, 'r') as f:
        phn_data[os.path.splitext(os.path.basename(phn_file))[0]] = f.read().splitlines()

# Extract speech tempo features
def extract_speech_tempo_features(audio_file, phn_data):
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000)

    # Load phonetic transcription
    utterance_id = os.path.splitext(os.path.basename(audio_file))[0]

    if utterance_id in phn_data:
        phones = phn_data[utterance_id]
    else:
        print(f"Skipping {utterance_id} due to missing phonetic transcriptions.")
        return None, None, None

    # Calculate speech rate
    syllables = sum(1 for phone in phones if phone[-3:] != 'cl')
    speech_duration = len(y) / sr
    speech_rate = syllables / speech_duration

    # Calculate pause frequency and duration
    pauses = [phone for phone in phones if phone[-3:] == 'cl']
    if len(pauses) != 0:
        pause_duration = sum(len(y) / sr for phone in pauses) / len(pauses)
        pause_frequency = len(pauses) / speech_duration
    else:
        pause_duration = -1
        pause_frequency = -1

    return speech_rate, pause_frequency, pause_duration

# Iterate through audio files and extract features
all_speech_rates = []
all_pause_frequencies = []
all_pause_durations = []

for audio_file in audio_files:
    speech_rate, pause_frequency, pause_duration = extract_speech_tempo_features(audio_file, phn_data)
    if speech_rate is not None:
        all_speech_rates.append(speech_rate)
        all_pause_frequencies.append(pause_frequency)
        all_pause_durations.append(pause_duration)

# Summarize the results
print(f"Average speech rate: {np.mean(all_speech_rates):.2f} syllables/second")
print(f"Average pause frequency: {np.mean(all_pause_frequencies):.2f} pauses/second")
print(f"Average pause duration: {np.mean(all_pause_durations):.2f} seconds")