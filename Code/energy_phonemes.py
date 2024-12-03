import os
import pandas as pd
import parselmouth
import numpy as np
# Set the speaker directory and mic type
speaker_dir = '/media/saketh/New Volume/SAL/Dataset/MC03/Session2'
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

# Extract phoneme-level energy features
def extract_phoneme_level_energy_features(audio_file, phn_data):
    # Load audio using Parselmouth
    sound = parselmouth.Sound(audio_file)
    
    # Load phonetic transcription
    utterance_id = os.path.splitext(os.path.basename(audio_file))[0]

    if utterance_id in phn_data:
        phones = phn_data[utterance_id]
    else:
        print(f"Skipping {utterance_id} due to missing phonetic transcriptions.")
        return None, None, None

    # Calculate phoneme-level energy features
    phoneme_energies = []
    phoneme_durations = []
    phones_with_energy = []
    for phone in phones:
        if phone != 'sil':
            # Calculate phoneme duration
            phoneme_start = sound.get_time_from_index(phones.index(phone))
            phoneme_end = sound.get_time_from_index(phones.index(phone) + 1)
            phoneme_duration = phoneme_end - phoneme_start
            phoneme_durations.append(phoneme_duration)

            # Calculate phoneme energy
            phoneme_segment = sound.extract_part(phoneme_start, phoneme_end)
            phoneme_energy = phoneme_segment.get_energy()
            phoneme_energies.append(phoneme_energy)
            phones_with_energy.append(phone)

    return phoneme_energies, phoneme_durations, phones_with_energy

# Iterate through audio files and extract phoneme-level energy features
all_phoneme_energies = []
all_phoneme_durations = []
all_phones = []

for audio_file in audio_files:
    phoneme_energies, phoneme_durations, phones = extract_phoneme_level_energy_features(audio_file, phn_data)
    if phoneme_energies is not None:
        all_phoneme_energies.extend(phoneme_energies)
        all_phoneme_durations.extend(phoneme_durations)
        all_phones.extend(phones)

# Calculate and display average energy for each unique phoneme in tabular format
phoneme_energy_dict = {}
phoneme_count_dict = {}
for phone in set(all_phones):
    phoneme_energies = [energy for i, energy in enumerate(all_phoneme_energies) if all_phones[i] == phone]
    phoneme_energy_dict[phone] = sum(phoneme_energies) / len(phoneme_energies)
    phoneme_count_dict[phone] = len(phoneme_energies)

# Determine the reference power for decibel conversion
reference_power = 1e-12  # Standard reference power for sound pressure level

# Create a DataFrame for average phoneme energies
phoneme_energy_list = sorted([
    {'Phoneme': phoneme, 
     'Average Energy (dB)': 10 * np.log10(phoneme_energy_dict[phoneme] / reference_power), 
     'Count': phoneme_count_dict[phoneme]} 
    for phoneme in phoneme_energy_dict
], key=lambda x: x['Phoneme'])

phoneme_energy_df = pd.DataFrame(phoneme_energy_list)
print("Average Energy for Each Phoneme:")
print(phoneme_energy_df.to_string(index=False))