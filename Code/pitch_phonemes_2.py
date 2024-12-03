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

# Extract phoneme-level pitch features
def extract_phoneme_level_pitch_features(audio_file, phn_data):
    # Load audio using Parselmouth
    sound = parselmouth.Sound(audio_file)
    
    # Load phonetic transcription
    utterance_id = os.path.splitext(os.path.basename(audio_file))[0]

    if utterance_id in phn_data:
        phones = phn_data[utterance_id]
    else:
        print(f"Skipping {utterance_id} due to missing phonetic transcriptions.")
        return None, None, None, None

    # Calculate phoneme-level pitch features
    phoneme_pitch_means = []
    phoneme_pitch_stds = []
    phoneme_pitch_ranges = []
    phones_with_pitch = []
    
    for phone in phones:
        if phone != 'sil':
            # Calculate phoneme duration
            phone_index = phones.index(phone)
            phoneme_start = sound.get_time_from_index(phone_index)
            phoneme_end = sound.get_time_from_index(phone_index + 1)
            
            # Extract pitch for the phoneme segment
            pitch = sound.to_pitch(
                time_step=0.01,  # 10 ms steps
                pitch_floor=75,  # Minimum pitch (Hz)
                pitch_ceiling=600  # Maximum pitch (Hz)
            )
            
            # Get pitch values within the phoneme segment
            pitch_values = []
            pitch_array = pitch.selected_array['frequency']
            time_array = pitch.xs()
            
            for t, pitch_value in zip(time_array, pitch_array):
                if phoneme_start <= t <= phoneme_end and pitch_value != 0:
                    pitch_values.append(pitch_value)
            
            # Only process phonemes with detectable pitch
            if pitch_values:
                phoneme_pitch_means.append(np.mean(pitch_values))
                phoneme_pitch_stds.append(np.std(pitch_values))
                phoneme_pitch_ranges.append(max(pitch_values) - min(pitch_values))
                phones_with_pitch.append(phone)

    return phoneme_pitch_means, phoneme_pitch_stds, phoneme_pitch_ranges, phones_with_pitch

# Iterate through audio files and extract phoneme-level pitch features
all_phoneme_pitch_means = []
all_phoneme_pitch_stds = []
all_phoneme_pitch_ranges = []
all_phones = []

for audio_file in audio_files:
    pitch_means, pitch_stds, pitch_ranges, phones = extract_phoneme_level_pitch_features(audio_file, phn_data)
    if pitch_means is not None:
        all_phoneme_pitch_means.extend(pitch_means)
        all_phoneme_pitch_stds.extend(pitch_stds)
        all_phoneme_pitch_ranges.extend(pitch_ranges)
        all_phones.extend(phones)

# Aggregate pitch features for each unique phoneme
phoneme_pitch_mean_dict = {}
phoneme_pitch_std_dict = {}
phoneme_pitch_range_dict = {}
phoneme_count_dict = {}

for phone in set(all_phones):
    # Filter pitch values for specific phoneme
    phone_pitch_means = [mean for i, mean in enumerate(all_phoneme_pitch_means) if all_phones[i] == phone]
    phone_pitch_stds = [std for i, std in enumerate(all_phoneme_pitch_stds) if all_phones[i] == phone]
    phone_pitch_ranges = [range_val for i, range_val in enumerate(all_phoneme_pitch_ranges) if all_phones[i] == phone]
    
    # Calculate aggregate statistics
    phoneme_pitch_mean_dict[phone] = np.mean(phone_pitch_means)
    phoneme_pitch_std_dict[phone] = np.mean(phone_pitch_stds)
    phoneme_pitch_range_dict[phone] = np.mean(phone_pitch_ranges)
    phoneme_count_dict[phone] = len(phone_pitch_means)

# Create a DataFrame for phoneme pitch features
phoneme_pitch_list = sorted([
    {
        'Phoneme': phoneme, 
        'Mean Pitch (Hz)': phoneme_pitch_mean_dict[phoneme], 
        'Pitch Std Dev (Hz)': phoneme_pitch_std_dict[phoneme],
        'Pitch Range (Hz)': phoneme_pitch_range_dict[phoneme],
        'Count': phoneme_count_dict[phoneme]
    } 
    for phoneme in phoneme_pitch_mean_dict
], key=lambda x: x['Phoneme'])

phoneme_pitch_df = pd.DataFrame(phoneme_pitch_list)
print("Pitch Characteristics for Each Phoneme:")
print(phoneme_pitch_df.to_string(index=False, float_format='{:.2f}'.format))