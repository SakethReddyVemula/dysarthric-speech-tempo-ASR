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
import pandas as pd

from dataclasses import dataclass
from typing import List, Dict, Tuple
import soundfile as sf
from typing import Dict, Tuple, Optional

import sox # type: ignore
import soundfile as sf

@dataclass
class PhonemeSegment:
    start: int
    end: int
    phoneme: str

def adjust_phoneme_segment_sox(audio: np.ndarray, 
                             start_sample: int,
                             end_sample: int,
                             tempo_ratio: float,
                             sr: int = 16000) -> np.ndarray:
    """
    Adjust tempo of a single phoneme segment using SoX
    
    Parameters:
    - audio: Audio signal
    - start_sample: Start sample of the phoneme
    - end_sample: End sample of the phoneme
    - tempo_ratio: Tempo adjustment ratio for the phoneme
    - sr: Sampling rate
    
    Returns:
    - Tempo-adjusted audio segment
    """
    # Create a temporary file for the segment
    temp_in = 'temp_in.wav'
    temp_out = 'temp_out.wav'
    
    # Extract and save segment
    segment = audio[start_sample:end_sample]
    sf.write(temp_in, segment, sr)
    
    # Create SoX transformer
    tfm = sox.Transformer()
    tfm.tempo(tempo_ratio)
    
    # Apply transformation
    tfm.build(temp_in, temp_out)
    
    # Read adjusted segment
    adjusted_segment, _ = librosa.load(temp_out, sr=sr)
    
    # Clean up temporary files
    os.remove(temp_in)
    os.remove(temp_out)
    
    return adjusted_segment

def phoneme_level_tempo_adjustment(audio_file: str,
                                 alignment_file: str,
                                 tempo_ratio_file: str,
                                 sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform phoneme-level tempo adjustment using SoX
    
    Parameters:
    - audio_file: Path to audio file
    - alignment_file: Path to phoneme alignment file
    - tempo_ratio_file: Path to tempo ratios JSON file
    - sr: Sampling rate
    
    Returns:
    - Tuple of (original_audio, adjusted_audio)
    """
    # Load audio
    audio, _ = librosa.load(audio_file, sr=sr)
    
    # Load phoneme alignment and tempo ratios
    phoneme_segments = load_phoneme_alignment(alignment_file)
    tempo_ratios = load_tempo_ratios(tempo_ratio_file)
    
    # Process each phoneme segment
    adjusted_segments = []
    
    for segment in phoneme_segments:
        # Get tempo ratio for current phoneme
        ratio = tempo_ratios.get(segment.phoneme, 1.0)
        
        # Extract and adjust segment using SoX
        adjusted_segment = adjust_phoneme_segment_sox(
            audio,
            segment.start,
            segment.end,
            ratio,
            sr
        )
        
        adjusted_segments.append(adjusted_segment)
    
    # Concatenate adjusted segments
    adjusted_audio = np.concatenate(adjusted_segments)
    
    return audio, adjusted_audio

def apply_sox_tempo_batch(audio_file: str, 
                         output_file: str,
                         tempo_ratio: float):
    """
    Apply SoX tempo adjustment to an entire audio file
    
    Parameters:
    - audio_file: Input audio file path
    - output_file: Output audio file path
    - tempo_ratio: Tempo adjustment ratio
    """
    tfm = sox.Transformer()
    tfm.tempo(tempo_ratio)
    tfm.build(audio_file, output_file)


def load_phoneme_alignment(alignment_file: str) -> List[PhonemeSegment]:
    """
    Load phoneme alignment from file
    
    Parameters:
    - alignment_file: Path to phoneme alignment file
    
    Returns:
    - List of PhonemeSegment objects
    """
    segments = []
    with open(alignment_file, 'r') as f:
        for line in f:
            start, end, phoneme = line.strip().split()
            segments.append(PhonemeSegment(
                start=int(start),
                end=int(end),
                phoneme=phoneme
            ))
    return segments

def load_tempo_ratios(ratio_file: str) -> Dict[str, float]:
    """
    Load tempo ratios from JSON file
    
    Parameters:
    - ratio_file: Path to JSON file containing tempo ratios
    
    Returns:
    - Dictionary mapping phonemes to their tempo ratios
    """
    with open(ratio_file, 'r') as f:
        return json.load(f)

def analyze_adjustment_results(original_audio: np.ndarray,
                             adjusted_audio: np.ndarray,
                             phoneme_segments: List[PhonemeSegment],
                             tempo_ratios: Dict[str, float],
                             sr: int = 16000) -> Dict:
    """
    Analyze the results of tempo adjustment
    
    Parameters:
    - original_audio: Original audio signal
    - adjusted_audio: Tempo-adjusted audio signal
    - phoneme_segments: List of PhonemeSegment segments
    - tempo_ratios: Dictionary of tempo ratios
    - sr: Sampling rate
    
    Returns:
    - Dictionary containing analysis results
    """
    results = {
        'original_duration': len(original_audio) / sr,
        'adjusted_duration': len(adjusted_audio) / sr,
        'phoneme_stats': {}
    }
    
    # Calculate per-phoneme statistics
    current_adjusted_position = 0
    
    for segment in phoneme_segments:
        phoneme = segment.phoneme
        if phoneme not in results['phoneme_stats']:
            results['phoneme_stats'][phoneme] = {
                'count': 0,
                'total_original_duration': 0,
                'total_adjusted_duration': 0,
                'tempo_ratio': tempo_ratios.get(phoneme, 1.0)
            }
        
        stats = results['phoneme_stats'][phoneme]
        original_duration = (segment.end - segment.start) / sr
        tempo_ratio = tempo_ratios.get(phoneme, 1.0)
        adjusted_duration = original_duration / tempo_ratio
        
        stats['count'] += 1
        stats['total_original_duration'] += original_duration
        stats['total_adjusted_duration'] += adjusted_duration
        
        # Update position tracker
        current_adjusted_position += adjusted_duration
    
    return results

def adjust_phoneme_segment(audio: np.ndarray, 
                         start_sample: int,
                         end_sample: int,
                         tempo_ratio: float,
                         sr: int = 16000) -> np.ndarray:
    """
    Adjust tempo of a single phoneme segment
    
    Parameters:
    - audio: Audio signal
    - start_sample: Start sample of the phoneme
    - end_sample: End sample of the phoneme
    - tempo_ratio: Tempo adjustment ratio for the phoneme
    - sr: Sampling rate
    
    Returns:
    - Tempo-adjusted audio segment
    """
    # Extract segment
    segment = audio[start_sample:end_sample]
    
    # Apply tempo adjustment using librosa
    adjusted_segment = librosa.effects.time_stretch(
        segment,
        rate=1/tempo_ratio  # Inverse ratio because librosa uses speed ratio
    )
    
    return adjusted_segment


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

def evaluate_phoneme_adjusted_speech(
    audio_file: str,
    phoneme_alignment_file: str,
    tempo_ratio_file: str,
    reference_text: Optional[str] = None,
    sr: int = 16000
) -> Dict:
    """
    Evaluate ASR performance on phoneme-level tempo-adjusted speech
    
    Parameters:
    - audio_file: Path to the audio file
    - phoneme_alignment_file: Path to phoneme alignment file
    - tempo_ratio_file: Path to tempo ratios JSON file
    - reference_text: Optional reference transcription
    - sr: Sampling rate
    
    Returns:
    - Dictionary containing evaluation results
    """
    # Load Wav2Vec2 model and processor
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    
    # Perform phoneme-level tempo adjustment
    original_audio, adjusted_audio = phoneme_level_tempo_adjustment(
        audio_file,
        phoneme_alignment_file,
        tempo_ratio_file,
        sr
    )
    
    # Process both original and adjusted audio through Wav2Vec2
    with torch.no_grad():
        # Original audio processing
        original_inputs = processor(
            original_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        original_outputs = model(original_inputs.input_values)
        
        # Adjusted audio processing
        adjusted_inputs = processor(
            adjusted_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        adjusted_outputs = model(adjusted_inputs.input_values)
    
    # Get transcriptions
    original_ids = torch.argmax(original_outputs.logits, dim=-1)
    adjusted_ids = torch.argmax(adjusted_outputs.logits, dim=-1)
    
    original_text = processor.decode(original_ids[0], skip_special_tokens=True)
    adjusted_text = processor.decode(adjusted_ids[0], skip_special_tokens=True)
    
    # Calculate confidence scores
    original_confidence = get_confidence_scores(original_outputs.logits)
    adjusted_confidence = get_confidence_scores(adjusted_outputs.logits)
    
    # Prepare results dictionary
    results = {
        'reference_text': reference_text,
        'original_transcription': original_text,
        'adjusted_transcription': adjusted_text,
        'original_confidence': original_confidence,
        'adjusted_confidence': adjusted_confidence,
        'original_duration': len(original_audio) / sr,
        'adjusted_duration': len(adjusted_audio) / sr,
    }
    
    # Calculate WER if reference text is provided
    if reference_text:
        results['original_wer'] = calculate_wer(reference_text, original_text)
        results['adjusted_wer'] = calculate_wer(reference_text, adjusted_text)
    
    # Add phoneme-level analysis
    phoneme_segments = load_phoneme_alignment(phoneme_alignment_file)
    tempo_ratios = load_tempo_ratios(tempo_ratio_file)
    phoneme_analysis = analyze_adjustment_results(
        original_audio,
        adjusted_audio,
        phoneme_segments,
        tempo_ratios,
        sr
    )
    results['phoneme_analysis'] = phoneme_analysis


    print(f"Referece Text: {reference_text}")
    print(f"Original Transcription: {original_text}")
    print(f"Adjusted_transcription: {adjusted_text}")

    
    return results

def visualize_results(results: Dict) -> None:
    """
    Visualize the evaluation results
    
    Parameters:
    - results: Dictionary containing evaluation results
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: WER comparison if available
    if 'original_wer' in results and 'adjusted_wer' in results:
        plt.subplot(2, 2, 1)
        wer_data = [results['original_wer'], results['adjusted_wer']]
        plt.bar(['Original', 'Tempo-adjusted'], wer_data)
        plt.title('Word Error Rate Comparison')
        plt.ylabel('WER')
    
    # Plot 2: Confidence scores
    plt.subplot(2, 2, 2)
    methods = ['Original', 'Tempo-adjusted']
    mean_conf = [results['original_confidence']['mean_confidence'],
                results['adjusted_confidence']['mean_confidence']]
    plt.bar(methods, mean_conf)
    plt.title('Mean Confidence Scores')
    plt.ylabel('Confidence')
    
    # Plot 3: Duration comparison
    plt.subplot(2, 2, 3)
    durations = [results['original_duration'], results['adjusted_duration']]
    plt.bar(methods, durations)
    plt.title('Audio Duration Comparison')
    plt.ylabel('Duration (seconds)')
    
    # Plot 4: Phoneme-level adjustments
    plt.subplot(2, 2, 4)
    phoneme_stats = results['phoneme_analysis']['phoneme_stats']
    common_phonemes = sorted(
        phoneme_stats.keys(),
        key=lambda x: phoneme_stats[x]['count'],
        reverse=True
    )[:10]  # Top 10 most frequent phonemes
    
    ratios = [phoneme_stats[p]['tempo_ratio'] for p in common_phonemes]
    plt.bar(common_phonemes, ratios)
    plt.title('Tempo Ratios for Most Common Phonemes')
    plt.xticks(rotation=45)
    plt.ylabel('Tempo Ratio')
    
    plt.tight_layout()
    plt.show()

def process_entire_dataset(base_dir, speaker, session, mic_type='headMic'):
    """
    Process entire dataset for a given speaker, session, and mic type.
    
    Parameters:
    - base_dir: Base directory of the dataset
    - speaker: Speaker identifier (e.g., 'M02')
    - session: Session identifier (e.g., 'Session1')
    - mic_type: Microphone type (default: 'arrayMic')
    
    Returns:
    - DataFrame with evaluation results
    """
    # Construct paths
    speaker_dir = os.path.join(base_dir, speaker, session)
    wav_dir = os.path.join(speaker_dir, f'wav_{mic_type}')
    phn_dir = os.path.join(speaker_dir, f'phn_{mic_type}')
    prompts_dir = os.path.join(speaker_dir, 'prompts')
    tempo_ratio_file = os.path.join('../utils', f'{speaker}_S2.json')
    output_dir = 'output_results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results storage
    results_list = []
    
    # Load tempo ratio data
    with open(tempo_ratio_file, 'r') as f:
        tempo_ratio_data = json.load(f)
    
    # Iterate through WAV files
    for wav_file in sorted(os.listdir(wav_dir)):
        if not wav_file.endswith('.wav'):
            continue
        
        # Construct full paths
        wav_path = os.path.join(wav_dir, wav_file)
        base_filename = os.path.splitext(wav_file)[0]
        
        # Corresponding phoneme alignment and prompt files
        phn_path = os.path.join(phn_dir, base_filename + '.PHN')
        prompt_path = os.path.join(prompts_dir, base_filename + '.txt')
        
        # Skip if any required file is missing
        if not (os.path.exists(phn_path) and os.path.exists(prompt_path)):
            continue
        
        # Load reference text
        with open(prompt_path, "r", encoding="utf-8") as file:
            reference_text = file.read().strip()  # Added strip() to remove leading/trailing whitespace
        
        # Skip if reference text starts with "[" or matches "*.jpg" pattern
        if reference_text.startswith('[') or reference_text.endswith('.jpg'):
            continue
            
        if reference_text.startswith("xxx"): # No speech signal in this
            continue

        # reference_text = reference_text.upper()
        reference_text = reference_text.lower()
        reference_text = reference_text.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        )
        reference_text = ' '.join(reference_text.split())
        
        try:
            # Run evaluation
            results = evaluate_phoneme_adjusted_speech(
                wav_path,
                phn_path,
                tempo_ratio_file,
                reference_text
            )
            
            # Store results
            results['wav_file'] = wav_file
            results_list.append(results)
        
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Compute average metrics
    avg_metrics = {
        'Average Original WER': results_df['original_wer'].mean(),
        'Average Adjusted WER': results_df['adjusted_wer'].mean(),
        'Average Original Duration': results_df['original_duration'].mean(),
        'Average Adjusted Duration': results_df['adjusted_duration'].mean()
    }
    
    # Save full results and summary
    results_df.to_csv(os.path.join(output_dir, f'{speaker}_{session}_wer_results.csv'), index=False)
    
    with open(os.path.join(output_dir, f'{speaker}_{session}_wer_summary.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    # Print summary
    print(f"\nMetrics for {speaker} - {session}:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return results_df, avg_metrics

def main():
    # Base dataset configuration
    base_dir = '../Dataset'
    speakers = ['M04']  # Add all speakers you want to process
    sessions = ['Session2']
    mic_types = ['headMic']
    
    # Store overall results
    overall_results = {}
    
    # Process each combination
    for speaker in speakers:
        for session in sessions:
            for mic_type in mic_types:
                print(f"\nProcessing: {speaker} - {session} - {mic_type}")
                try:
                    results_df, avg_metrics = process_entire_dataset(

                        base_dir, 
                        speaker, 
                        session, 
                        mic_type
                    )
                    
                    # Store results
                    key = f"{speaker}_{session}_{mic_type}"
                    overall_results[key] = avg_metrics
                
                except Exception as e:
                    print(f"Error processing {speaker} - {session} - {mic_type}: {e}")
    
    # Save overall results
    with open('overall_wer_summary.json', 'w') as f:
        json.dump(overall_results, f, indent=4)

if __name__ == "__main__":
    main()