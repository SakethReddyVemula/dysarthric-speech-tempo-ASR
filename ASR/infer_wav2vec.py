import torch
from transformers import AutoModelForCTC, AutoProcessor
import librosa
import numpy as np
import editdistance

def calculate_wer(audio_paths, reference_transcripts, model_name='facebook/wav2vec2-base-960h'):
    """
    Calculate Word Error Rate (WER) using a Wav2Vec model with manual WER calculation
    
    Args:
        audio_paths (list): List of paths to audio files
        reference_transcripts (list): Corresponding ground truth transcriptions
        model_name (str): Hugging Face model name to use
    
    Returns:
        dict: Detailed WER metrics
    """
    # Load pre-trained model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Lists to store results
    predicted_transcripts = []
    wer_scores = []
    
    # Process each audio file
    for audio_path, reference in zip(audio_paths, reference_transcripts):
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Preprocess audio
        input_values = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode predicted ids to text
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcript = processor.batch_decode(predicted_ids)[0]
        
        # Preprocess reference and predicted transcripts
        ref_words = _preprocess_text(str(reference))
        pred_words = _preprocess_text(predicted_transcript)
        
        # Calculate Word Error Rate manually
        wer = _calculate_wer(ref_words, pred_words)
        
        predicted_transcripts.append(predicted_transcript)
        wer_scores.append(wer)
    
    # Calculate overall metrics
    avg_wer = np.mean(wer_scores)
    
    return {
        'predicted_transcripts': predicted_transcripts,
        'wer_scores': wer_scores,
        'average_wer': avg_wer
    }

def _preprocess_text(text):
    """
    Preprocess text by converting to lowercase and splitting into words
    
    Args:
        text (str): Input text
    
    Returns:
        list: Processed words
    """
    # Convert to lowercase, remove punctuation, split into words
    import string
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words and remove empty strings
    return [word.strip() for word in text.split() if word.strip()]

def _calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate manually using edit distance
    
    Args:
        reference (list): Reference words
        hypothesis (list): Hypothesis words
    
    Returns:
        float: Word Error Rate
    """
    # Calculate edit distance
    distance = editdistance.eval(reference, hypothesis)
    
    # Calculate WER
    wer = distance / len(reference) if reference else 0.0
    
    return wer

# Example usage
def main():
    # Replace these with your actual audio paths and transcriptions
    # audio_paths = [
    #     '/media/saketh/New Volume/SAL/Dataset/MC01/Session1/wav_headMic/0016.wav', 
    #     '/media/saketh/New Volume/SAL/Dataset/MC01/Session1/wav_headMic/0019.wav'
    # ]
    # reference_transcripts = [
    #     'You wished to know all about my grandfather.', 
    #     'yet he still thinks as swiftly as ever.'
    # ]

    audio_paths = [
        '/media/saketh/New Volume/SAL/Dataset/M01/Session1/wav_headMic/0019.wav', 
    ]
    reference_transcripts = [
        'he dresses himself in an ancient black frock coat, ', 
    ]
    # Calculate WER
    results = calculate_wer(audio_paths, reference_transcripts)
    
    # Print results
    print("Transcription Comparison:")
    for ref, pred, wer in zip(
        reference_transcripts, 
        results['predicted_transcripts'],
        results['wer_scores']
    ):
        print(f"Reference: {ref}")
        print(f"Predicted: {pred}")
        print(f"WER: {wer:.4f}\n")
    
    print(f"Average WER: {results['average_wer']:.4f}")

if __name__ == '__main__':
    main()