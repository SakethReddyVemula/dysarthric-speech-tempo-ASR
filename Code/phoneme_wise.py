import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import defaultdict

# Set the speaker directory and mic type
speaker_dir = '/media/saketh/New Volume/SAL/Dataset/M01/Session2_3'
mic_type = 'arrayMic'  # Change to 'headMic' for wav_headMic/phn_headMic

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
        phn_data[os.path.splitext(os.path.basename(phn_file))[0]] = [line.split()[2] for line in f.read().splitlines()]

# Extract phoneme-level speech tempo features
def extract_phoneme_level_speech_tempo_features(audio_file, phn_data):
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000)

    # Load phonetic transcription
    utterance_id = os.path.splitext(os.path.basename(audio_file))[0]

    if utterance_id in phn_data:
        phones = phn_data[utterance_id]
    else:
        print(f"Skipping {utterance_id} due to missing phonetic transcriptions.")
        return None, None, None

    # Calculate phoneme-level features
    phoneme_durations = []
    phoneme_features = []
    for phone in phones:
        if phone != 'sil':
            # Calculate phoneme duration (assuming uniform duration)
            phoneme_duration = len(y) / (sr * len(phones))
            phoneme_durations.append(phoneme_duration)

            # Calculate phoneme-level speech rate
            phoneme_syllables = 1 if phone[-3:] not in ['ae', 'ao'] else 2
            phoneme_speech_rate = phoneme_syllables / phoneme_duration
            phoneme_features.append(phoneme_speech_rate)

    return phoneme_durations, phoneme_features, phones

# Iterate through audio files and extract phoneme-level features
all_phoneme_durations = []
all_phoneme_speech_rates = []
all_phones = []

for audio_file in audio_files:
    phoneme_durations, phoneme_speech_rates, phones = extract_phoneme_level_speech_tempo_features(audio_file, phn_data)
    if phoneme_durations is not None:
        all_phoneme_durations.extend(phoneme_durations)
        all_phoneme_speech_rates.extend(phoneme_speech_rates)
        all_phones.extend(phones)

# Summarize the results in tabular format
summary_data = {
    'Average Phoneme Duration (seconds)': [np.mean(all_phoneme_durations)],
    'Average Phoneme-level Speech Rate (syllables/second)': [np.mean(all_phoneme_speech_rates)]
}
summary_df = pd.DataFrame(summary_data)
print("Summary Statistics:")
print(summary_df)

# Calculate and display average duration for each unique phoneme in tabular format
phoneme_duration_dict = defaultdict(list)
phoneme_count_dict = defaultdict(int)
for i, phone in enumerate(all_phones):
    if phone != 'sil':
        for j, duration in enumerate(all_phoneme_durations):
            if i == j:
                phoneme_duration_dict[phone].append(duration)
                phoneme_count_dict[phone] += 1
                break

# Create a DataFrame for average phoneme durations
phoneme_duration_list = sorted([
    {'Phoneme': phoneme, 
     'Average Duration (seconds)': np.mean(durations), 
     'Count': phoneme_count_dict[phoneme]} 
    for phoneme, durations in phoneme_duration_dict.items()
], key=lambda x: x['Phoneme'])

phoneme_duration_df = pd.DataFrame(phoneme_duration_list)
print("\nAverage Duration for Each Phoneme:")
print(phoneme_duration_df.to_string(index=False))
