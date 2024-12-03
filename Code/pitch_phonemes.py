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

# Load phonetic transcriptions with start and end times
def load_phonetic_transcriptions(phn_file):
    phn_data = []
    with open(phn_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start_time = float(parts[0]) / 16000  # Convert sample numbers to seconds
            end_time = float(parts[1]) / 16000
            phoneme = parts[2]
            phn_data.append({
                'start': start_time,
                'end': end_time,
                'phoneme': phoneme
            })
    return phn_data

# Load phonetic transcriptions
phn_data = {}
dir_path = os.path.join(speaker_dir, phn_dir)
phn_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.PHN') or f.endswith('.phn')]
for phn_file in phn_files:
    utterance_id = os.path.splitext(os.path.basename(phn_file))[0]
    phn_data[utterance_id] = load_phonetic_transcriptions(phn_file)

# Extract phoneme-level pitch features with improved robustness
def extract_phoneme_level_pitch_features(audio_file, phn_data):
    try:
        # Load audio using Parselmouth
        sound = parselmouth.Sound(audio_file)
        
        # Load phonetic transcription
        utterance_id = os.path.splitext(os.path.basename(audio_file))[0]

        if utterance_id not in phn_data:
            print(f"Skipping {utterance_id} due to missing phonetic transcriptions.")
            return None, None, None, None

        # Calculate phoneme-level pitch features
        phoneme_pitch_means = []
        phoneme_pitch_stds = []
        phoneme_durations = []
        phones_with_pitch = []

        for phone_info in phn_data[utterance_id]:
            # Skip silence and very short segments
            if phone_info['phoneme'] == 'sil' or (phone_info['end'] - phone_info['start']) < 0.01:
                continue

            try:
                # Extract phoneme segment
                start_time = phone_info['start']
                end_time = phone_info['end']
                phoneme_duration = end_time - start_time
                phoneme_segment = sound.extract_part(start_time, end_time)

                # Dynamically adjust pitch analysis parameters
                pitch_floor = max(30, min(100, phoneme_duration * 10))
                pitch_ceiling = min(300, max(200, phoneme_duration * 50))

                # Extract pitch with dynamic parameters
                pitch = phoneme_segment.to_pitch(
                    time_step=0.01,  # 10 ms time step
                    pitch_floor=pitch_floor,  # Dynamically adjusted minimum pitch
                    pitch_ceiling=pitch_ceiling  # Dynamically adjusted maximum pitch
                )
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames

                if len(pitch_values) > 0:
                    phoneme_pitch_means.append(np.mean(pitch_values))
                    phoneme_pitch_stds.append(np.std(pitch_values))
                    phoneme_durations.append(phoneme_duration)
                    phones_with_pitch.append(phone_info['phoneme'])
                else:
                    # Fallback for very short or unvoiced segments
                    print(f"No valid pitch values for {phone_info['phoneme']} segment")
            
            except Exception as phone_error:
                print(f"Error processing phone {phone_info['phoneme']}: {phone_error}")

        return phoneme_pitch_means, phoneme_pitch_stds, phoneme_durations, phones_with_pitch

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None, None, None, None

# Iterate through audio files and extract phoneme-level pitch features
all_phoneme_pitch_means = []
all_phoneme_pitch_stds = []
all_phoneme_durations = []
all_phones = []

for audio_file in audio_files:
    phoneme_pitch_means, phoneme_pitch_stds, phoneme_durations, phones = extract_phoneme_level_pitch_features(audio_file, phn_data)
    
    if phoneme_pitch_means is not None:
        all_phoneme_pitch_means.extend(phoneme_pitch_means)
        all_phoneme_pitch_stds.extend(phoneme_pitch_stds)
        all_phoneme_durations.extend(phoneme_durations)
        all_phones.extend(phones)

# Calculate and display pitch statistics for each unique phoneme
phoneme_pitch_mean_dict = {}
phoneme_pitch_std_dict = {}
phoneme_count_dict = {}

for phone in set(all_phones):
    # Filter pitch values for specific phoneme
    phoneme_pitch_means = [pitch for i, pitch in enumerate(all_phoneme_pitch_means) if all_phones[i] == phone]
    phoneme_pitch_stds = [std for i, std in enumerate(all_phoneme_pitch_stds) if all_phones[i] == phone]

    # Calculate mean and standard deviation
    phoneme_pitch_mean_dict[phone] = np.mean(phoneme_pitch_means)
    phoneme_pitch_std_dict[phone] = np.mean(phoneme_pitch_stds)
    phoneme_count_dict[phone] = len(phoneme_pitch_means)

# Create a DataFrame for pitch statistics
phoneme_pitch_list = sorted([
    {
        'Phoneme': phoneme, 
        'Mean Pitch (Hz)': phoneme_pitch_mean_dict[phoneme], 
        'Pitch Variability (Hz)': phoneme_pitch_std_dict[phoneme],
        'Count': phoneme_count_dict[phoneme]
    } 
    for phoneme in phoneme_pitch_mean_dict
], key=lambda x: x['Phoneme'])

phoneme_pitch_df = pd.DataFrame(phoneme_pitch_list)
print("Pitch Statistics for Each Phoneme:")
print(phoneme_pitch_df.to_string(index=False, float_format='{:.2f}'.format))