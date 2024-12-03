import os
import numpy as np
import scipy.signal as signal
import librosa
import scipy.interpolate as interpolate
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer # type: ignore
import torch.nn.functional as F
import string
import json

def tempo_adjust_mfcc(mfcc_features, tempo_ratio):
    """
    Adjust MFCC features based on tempo ratio
    
    Parameters:
    - mfcc_features: Original MFCC features (time x features)
    - tempo_ratio: Ratio to adjust tempo (e.g., 1.2 for 20% faster, 0.8 for 20% slower)
    
    Returns:
    - Tempo-adjusted MFCC features
    """
    # Create time indices
    original_time = np.arange(mfcc_features.shape[0])
    
    # Adjust time indices based on tempo ratio
    adjusted_time = original_time / tempo_ratio
    
    # Interpolate to create new feature representation
    interpolator = interpolate.interp1d(
        original_time, 
        mfcc_features.T,  # Transpose for interpolation
        kind='linear',
        fill_value='extrapolate'
    )
    
    # Resample features
    adjusted_mfcc = interpolator(adjusted_time).T
    
    return adjusted_mfcc

def apply_phoneme_level_tempo_adjustment(mfcc_features, phoneme_tempo_ratios):
    """
    Apply phoneme-level tempo adjustments
    
    Parameters:
    - mfcc_features: Original MFCC features
    - phoneme_tempo_ratios: Dictionary of phoneme-level tempo ratios
    
    Returns:
    - Tempo-adjusted MFCC features
    """
    # Placeholder for more complex phoneme-level adjustment
    # This would require phoneme alignment information
    # You'd segment MFCC features based on phoneme boundaries
    # and apply individual tempo ratios to each segment
    
    # For now, a simplified approach
    adjusted_features = mfcc_features.copy()
    
    # Example of how you might apply phoneme-level adjustments
    for phoneme, ratio in phoneme_tempo_ratios.items():
        # Identify phoneme segments (you'll need phoneme alignment logic)
        # Apply tempo adjustment to those specific segments
        pass
    
    return adjusted_features

def load_test_mfcc(test_audio_dir, test_tempo_ratios):
    # Load test MFCC features from your data source
    test_mfcc = get_test_mfcc_features(test_audio_dir, test_tempo_ratios)
    return test_mfcc

def infer_with_wav2vec2(test_mfcc, test_tempo_ratios):
    """
    Perform inference using pre-trained Wav2Vec2 model on tempo-adjusted test data
    
    Parameters:
    - test_mfcc: MFCC features for test data
    - test_tempo_ratios: Tempo ratios for each test sample
    
    Returns:
    - Predicted text transcripts
    """
    # Load pre-trained Wav2Vec2 model and processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Apply tempo adjustment to test MFCC features
    for sample_idx, tempo_ratio in enumerate(test_tempo_ratios):
        test_mfcc[sample_idx] = tempo_adjust_mfcc(test_mfcc[sample_idx], tempo_ratio)
    
    # Prepare input features
    input_features = processor(test_mfcc, return_tensors="pt", padding=True).input_features
    
    # Generate predictions
    output_logits = model(input_features).logits
    predicted_ids = torch.argmax(output_logits, dim=-1)
    
    # Decode predictions
    predicted_text = [processor.decode(predicted_id, skip_special_tokens=True) for predicted_id in predicted_ids]
    
    return predicted_text

def extract_mfcc_features(audio_file):
    """
    Extract MFCC features from a given audio file.
    
    Parameters:
    - audio_file (str): Path to the audio file.
    
    Returns:
    - mfcc_features (np.ndarray): MFCC features extracted from the audio.
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    return mfcc.T  # Return transposed MFCC features (time x features)

def get_test_mfcc_features(test_audio_dir, test_tempo_ratios):
    """
    Extract MFCC features for all test audio files and apply tempo adjustment.
    
    Parameters:
    - test_audio_dir (str): Path to the directory containing test audio files.
    - test_tempo_ratios (list): List of tempo ratios for each test sample.
    
    Returns:
    - test_mfcc (np.ndarray): Tempo-adjusted MFCC features for test data.
    """
    test_mfcc = []
    
    for audio_file, tempo_ratio in zip(os.listdir(test_audio_dir), test_tempo_ratios):
        audio_path = os.path.join(test_audio_dir, audio_file)
        mfcc_features = extract_mfcc_features(audio_path)
        
        # Apply tempo adjustment
        adjusted_mfcc = tempo_adjust_mfcc(mfcc_features, tempo_ratio)
        test_mfcc.append(adjusted_mfcc)
    
    return np.array(test_mfcc)

def test_single_sample(audio_file_path, tempo_ratio):
    """
    Test tempo-adjusted ASR on a single audio sample
    
    Parameters:
    - audio_file_path (str): Path to the dysarthric speech audio file
    - tempo_ratio (float): Tempo adjustment ratio (e.g., 0.8 for slower, 1.2 for faster)
    
    Returns:
    - dict: Dictionary containing original and tempo-adjusted transcriptions
    """
    import librosa
    import numpy as np
    
    # Load Wav2Vec2 model and processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Load the audio file
    audio, sr = librosa.load(audio_file_path, sr=16000)  # Wav2Vec2 expects 16kHz
    
    # Extract MFCC features for tempo adjustment
    original_mfcc = extract_mfcc_features(audio_file_path)
    adjusted_mfcc = tempo_adjust_mfcc(original_mfcc, tempo_ratio)
    
    # Calculate the length ratio between original and adjusted MFCCs
    length_ratio = len(adjusted_mfcc) / len(original_mfcc)
    
    # Adjust the raw audio length to match the MFCC adjustment
    target_length = int(len(audio) * length_ratio)
    adjusted_audio = librosa.effects.time_stretch(audio, rate=1/tempo_ratio)
    
    # Ensure both audios are properly processed by the model
    original_inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    adjusted_inputs = processor(adjusted_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Generate predictions for original speech
    with torch.no_grad():
        original_logits = model(original_inputs.input_values).logits
        adjusted_logits = model(adjusted_inputs.input_values).logits
    
    original_ids = torch.argmax(original_logits, dim=-1)
    adjusted_ids = torch.argmax(adjusted_logits, dim=-1)
    
    original_text = processor.decode(original_ids[0], skip_special_tokens=True)
    adjusted_text = processor.decode(adjusted_ids[0], skip_special_tokens=True)
    
    results = {
        'original_transcription': original_text,
        'tempo_adjusted_transcription': adjusted_text,
        'tempo_ratio': tempo_ratio,
        'original_audio_length': len(audio),
        'adjusted_audio_length': len(adjusted_audio),
        'original_mfcc_shape': original_mfcc.shape,
        'adjusted_mfcc_shape': adjusted_mfcc.shape
    }
    
    return results

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis transcriptions
    
    Parameters:
    - reference (str): Reference transcription
    - hypothesis (str): Hypothesis transcription
    
    Returns:
    - float: Word Error Rate
    """
    return wer(reference, hypothesis)

def get_confidence_scores(logits):
    """
    Calculate confidence scores from model logits
    
    Parameters:
    - logits (torch.Tensor): Model output logits
    
    Returns:
    - dict: Dictionary containing mean, min, and max confidence scores
    """
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get the maximum probability for each prediction
    confidence_scores = torch.max(probs, dim=-1)[0]
    
    return {
        'mean_confidence': float(torch.mean(confidence_scores).item()),
        'min_confidence': float(torch.min(confidence_scores).item()),
        'max_confidence': float(torch.max(confidence_scores).item())
    }

def visualize_mfcc_comparison(original_mfcc, adjusted_mfcc, tempo_ratio):
    """
    Create visualization comparing original and tempo-adjusted MFCC features
    
    Parameters:
    - original_mfcc (np.ndarray): Original MFCC features
    - adjusted_mfcc (np.ndarray): Tempo-adjusted MFCC features
    - tempo_ratio (float): Tempo adjustment ratio
    
    Returns:
    - matplotlib.figure.Figure: Figure containing the visualization
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original MFCC
    im1 = ax1.imshow(original_mfcc.T, aspect='auto', origin='lower')
    ax1.set_title('Original MFCC Features')
    ax1.set_xlabel('Time Frames')
    ax1.set_ylabel('MFCC Coefficients')
    plt.colorbar(im1, ax=ax1)
    
    # Plot adjusted MFCC
    im2 = ax2.imshow(adjusted_mfcc.T, aspect='auto', origin='lower')
    ax2.set_title(f'Tempo-Adjusted MFCC Features (ratio: {tempo_ratio})')
    ax2.set_xlabel('Time Frames')
    ax2.set_ylabel('MFCC Coefficients')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig

def enhanced_test_single_sample(audio_file_path, tempo_ratio, reference_text=None):
    """
    Enhanced version of test_single_sample with additional analysis features
    
    Parameters:
    - audio_file_path (str): Path to the dysarthric speech audio file
    - tempo_ratio (float): Tempo adjustment ratio
    - reference_text (str, optional): Reference transcription for WER calculation
    
    Returns:
    - dict: Dictionary containing transcriptions and analysis results
    """
    import librosa
    
    # Load Wav2Vec2 model and processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Load the audio file
    audio, sr = librosa.load(audio_file_path, sr=16000)
    
    # Extract MFCC features
    original_mfcc = extract_mfcc_features(audio_file_path)
    adjusted_mfcc = tempo_adjust_mfcc(original_mfcc, tempo_ratio)
    
    # Calculate the length ratio between original and adjusted MFCCs
    length_ratio = len(adjusted_mfcc) / len(original_mfcc)
    
    # Adjust the raw audio length
    target_length = int(len(audio) * length_ratio)
    adjusted_audio = librosa.effects.time_stretch(audio, rate=1/tempo_ratio)
    
    # Process audio through Wav2Vec2
    original_inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    adjusted_inputs = processor(adjusted_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Generate predictions
    with torch.no_grad():
        original_outputs = model(original_inputs.input_values)
        adjusted_outputs = model(adjusted_inputs.input_values)
    
    original_logits = original_outputs.logits
    adjusted_logits = adjusted_outputs.logits
    
    # Get predicted IDs and transcriptions
    original_ids = torch.argmax(original_logits, dim=-1)
    adjusted_ids = torch.argmax(adjusted_logits, dim=-1)
    
    original_text = processor.decode(original_ids[0], skip_special_tokens=True)
    adjusted_text = processor.decode(adjusted_ids[0], skip_special_tokens=True)
    
    # Calculate confidence scores
    original_confidence = get_confidence_scores(original_logits)
    adjusted_confidence = get_confidence_scores(adjusted_logits)
    
    # Create visualization
    mfcc_visualization = visualize_mfcc_comparison(original_mfcc, adjusted_mfcc, tempo_ratio)
    
    results = {
        'original_transcription': original_text,
        'tempo_adjusted_transcription': adjusted_text,
        'tempo_ratio': tempo_ratio,
        'original_confidence': original_confidence,
        'adjusted_confidence': adjusted_confidence,
        'original_audio_length': len(audio),
        'adjusted_audio_length': len(adjusted_audio),
        'original_mfcc_shape': original_mfcc.shape,
        'adjusted_mfcc_shape': adjusted_mfcc.shape,
        'mfcc_visualization': mfcc_visualization
    }
    
    # Calculate WER if reference text is provided
    if reference_text is not None:
        results['original_wer'] = calculate_wer(reference_text, original_text)
        results['adjusted_wer'] = calculate_wer(reference_text, adjusted_text)
    
    return results


# Example usage
audio_file = "/media/saketh/New Volume/SAL/Dataset/M02/Session1/wav_arrayMic/0023.wav"
phoneme_alignment_file = "/media/saketh/New Volume/SAL/Dataset/M02/Session1/phn_arrayMic/0023.PHN"
prompt_file = "/media/saketh/New Volume/SAL/Dataset/M02/Session1/prompts/0023.txt"
tempo_ratio_file = "/media/saketh/New Volume/SAL/utils/M02_S1.json"

with open(tempo_ratio_file, "r", encoding="utf-8") as file:
    tempo_ratios = json.load(file)

tempo_ratio = tempo_ratios["Total/Avg"]
print(f"tempo_ratio: {tempo_ratio}")

# reference_text = "WELL HE IS NEARLY NINETY THREE YEARS OLD"
with open(prompt_file, "r", encoding="utf-8") as file:
    reference_text = file.read()

if reference_text.startswith("["):
    print("reference text is a prompt")
reference_text = reference_text.upper()
reference_text = reference_text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
reference_text = ' '.join(reference_text.split())
print(reference_text)


results = enhanced_test_single_sample(audio_file, tempo_ratio, reference_text)

# Print transcriptions and WER
print("Original Transcription:", results['original_transcription'])
print("Tempo-adjusted Transcription:", results['tempo_adjusted_transcription'])
if reference_text:
    print("Original WER:", results['original_wer'])
    print("Adjusted WER:", results['adjusted_wer'])

# Print confidence scores
print("\nOriginal Confidence Scores:")
print(results['original_confidence'])
print("\nAdjusted Confidence Scores:")
print(results['adjusted_confidence'])

# Display MFCC visualization
plt.show()  # This will display the MFCC comparison plot