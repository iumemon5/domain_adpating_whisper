#!/usr/bin/env python3
"""
Example: Testing Script
This example shows how to test the domain-adapted Whisper model with different configurations.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from test_whisper_model import (
    load_model_and_processor,
    test_single_file,
    test_batch_files,
    compute_metrics
)

def example_single_file_testing():
    """Example of testing a single audio file."""
    print("="*60)
    print("EXAMPLE: Single File Testing")
    print("="*60)
    
    # Example command line usage
    print("Command line usage:")
    print("python test_whisper_model.py --test-file audio.wav --domain dyslexic --verbose")
    print()
    
    # Programmatic usage
    print("Programmatic usage:")
    print("""
# Load model
model, processor = load_model_and_processor("runs/whisper_dyslexia_domain_adaptation/checkpoint-1000")

# Test single file
result = test_single_file(
    model=model,
    processor=processor,
    audio_file="path/to/audio.wav",
    reference_text="Expected transcription",
    domain="dyslexic"
)

print(f"Transcription: {result['transcription']}")
print(f"CER: {result['cer']:.4f}")
print(f"WER: {result['wer']:.4f}")
""")

def example_batch_testing():
    """Example of batch testing with JSON configuration."""
    print("="*60)
    print("EXAMPLE: Batch Testing")
    print("="*60)
    
    # Create example test data
    example_data = [
        {
            "audio_path": "examples/sample_audio_1.wav",
            "text": "안녕하세요, 이것은 테스트입니다.",
            "domain": "normal"
        },
        {
            "audio_path": "examples/sample_audio_2.wav", 
            "text": "한국어 음성 인식 모델을 테스트하고 있습니다.",
            "domain": "dyslexic"
        }
    ]
    
    # Save example data
    with open("examples/sample_test_data.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    print("Created example test data: examples/sample_test_data.json")
    print()
    
    print("Command line usage:")
    print("python test_whisper_model.py --test-json examples/sample_test_data.json --verbose --output results.json")
    print()
    
    print("Programmatic usage:")
    print("""
# Load model
model, processor = load_model_and_processor("runs/whisper_dyslexia_domain_adaptation/checkpoint-1000")

# Load test data
with open("examples/sample_test_data.json", "r") as f:
    test_data = json.load(f)

audio_files = [item["audio_path"] for item in test_data]
reference_texts = [item["text"] for item in test_data]

# Test batch
results = test_batch_files(
    model=model,
    processor=processor,
    test_files=audio_files,
    reference_texts=reference_texts,
    domain="normal"  # or "dyslexic"
)

# Compute overall metrics
predictions = [r["transcription"] for r in results if r["transcription"]]
references = [r["reference"] for r in results if r["reference"]]
metrics = compute_metrics(predictions, references)

print(f"Overall CER: {metrics['cer']:.4f}")
print(f"Overall WER: {metrics['wer']:.4f}")
""")

def example_domain_comparison():
    """Example of comparing normal vs dyslexic domain performance."""
    print("="*60)
    print("EXAMPLE: Domain Comparison")
    print("="*60)
    
    print("To compare performance across domains:")
    print()
    print("# Test with normal domain")
    print("python test_whisper_model.py --test-file audio.wav --domain normal --verbose")
    print()
    print("# Test with dyslexic domain") 
    print("python test_whisper_model.py --test-file audio.wav --domain dyslexic --verbose")
    print()
    
    print("Programmatic comparison:")
    print("""
# Load model
model, processor = load_model_and_processor("checkpoint_path")

# Test same audio with both domains
audio_file = "path/to/audio.wav"
reference_text = "Expected transcription"

# Normal domain
result_normal = test_single_file(
    model, processor, audio_file, reference_text, domain="normal"
)

# Dyslexic domain  
result_dyslexic = test_single_file(
    model, processor, audio_file, reference_text, domain="dyslexic"
)

print(f"Normal domain - CER: {result_normal['cer']:.4f}")
print(f"Dyslexic domain - CER: {result_dyslexic['cer']:.4f}")
print(f"Improvement: {((result_normal['cer'] - result_dyslexic['cer']) / result_normal['cer'] * 100):.1f}%")
""")

def example_dataset_evaluation():
    """Example of evaluating on the full dyslexia dataset."""
    print("="*60)
    print("EXAMPLE: Dataset Evaluation")
    print("="*60)
    
    print("Command line usage:")
    print("python test_dyslexia_dataset.py --dataset-root /path/to/dataset --verbose --output evaluation.json")
    print()
    
    print("This will:")
    print("- Load all test files from the dyslexia dataset")
    print("- Test with appropriate domain settings")
    print("- Compute comprehensive metrics")
    print("- Generate detailed evaluation report")
    print("- Save results to JSON file")
    print()
    
    print("Advanced options:")
    print("--max-files 100  # Limit to first 100 files for quick testing")
    print("--use-webdataset  # Use webdataset format")
    print("--use-traditional  # Use traditional directory format")

def example_error_handling():
    """Example of proper error handling in testing."""
    print("="*60)
    print("EXAMPLE: Error Handling")
    print("="*60)
    
    print("""
# Robust testing with error handling
def safe_test_file(model, processor, audio_file, reference_text=None):
    try:
        result = test_single_file(model, processor, audio_file, reference_text)
        
        if result.get('error'):
            print(f"Error processing {audio_file}: {result['error']}")
            return None
            
        return result
        
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file}")
        return None
    except Exception as e:
        print(f"Unexpected error with {audio_file}: {e}")
        return None

# Usage
result = safe_test_file(model, processor, "audio.wav", "reference text")
if result:
    print(f"Success: CER = {result['cer']:.4f}")
""")

if __name__ == "__main__":
    example_single_file_testing()
    print()
    example_batch_testing()
    print()
    example_domain_comparison()
    print()
    example_dataset_evaluation()
    print()
    example_error_handling()
    
    print("="*60)
    print("Testing Examples Summary:")
    print("1. Single file testing for quick validation")
    print("2. Batch testing for comprehensive evaluation")
    print("3. Domain comparison for performance analysis")
    print("4. Dataset evaluation for full assessment")
    print("5. Error handling for robust testing")
    print("="*60)
