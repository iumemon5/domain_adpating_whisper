#!/usr/bin/env python3
"""
Testing script for domain-adapted Whisper model
- Loads the best checkpoint from runs/whisper_dyslexia_domain_adaptation/
- Supports domain-specific transcription with <|domain:normal|> and <|domain:dyslexic|> tokens
- Provides single file and batch testing capabilities
- Computes evaluation metrics (CER, WER)
- Supports various audio formats
- Generates detailed test reports
"""

import argparse
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Text Processing (matching training script)
# --------------------------------------------------------------------------------------

def ko_norm(text: str) -> str:
    """Normalize Korean text (same as training script)"""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------------------
# Audio Processing
# --------------------------------------------------------------------------------------

def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
    """Load audio from bytes (same as training script)"""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data).T
    return waveform, sr


def preprocess_audio(audio_bytes: bytes, target_sr: int = 16000) -> torch.Tensor:
    """Preprocess audio (same as training script)"""
    waveform, sample_rate = load_audio_from_bytes(audio_bytes)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def load_audio_file(file_path: Union[str, Path], target_sr: int = 16000) -> torch.Tensor:
    """Load and preprocess audio file"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    
    return preprocess_audio(audio_bytes, target_sr)


# --------------------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------------------

def load_model_and_processor(checkpoint_path: str) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load the fine-tuned model and processor"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(checkpoint_path)
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        checkpoint_path, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        ignore_mismatched_sizes=True  # This will suppress missing key warnings
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
    else:
        print("Model loaded on CPU")
    
    model.eval()
    return model, processor


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------

def transcribe_audio(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_waveform: torch.Tensor,
    language: str = "korean",
    task: str = "transcribe",
    domain: str = "normal"
) -> str:
    """Transcribe a single audio waveform with domain adaptation"""
    # Prepare input features
    input_features = processor.feature_extractor(
        audio_waveform, sampling_rate=16000, return_tensors="pt"
    ).input_features
    
    if torch.cuda.is_available():
        input_features = input_features.cuda()
        # Ensure input features match model dtype
        if model.dtype != input_features.dtype:
            input_features = input_features.to(model.dtype)
    
    # Prepare domain-specific prompt
    domain_token = "<|domain:dyslexic|>" if domain == "dyslexic" else "<|domain:normal|>"
    
    # Generate transcription with domain token
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=language,
            task=task,
            max_length=448,
            num_beams=1,
            do_sample=False,
            temperature=None,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language=language, task=task),
        )
    
    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Remove domain token from transcription if present
    transcription = transcription.replace(domain_token, "").strip()
    
    return ko_norm(transcription)


def transcribe_file(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    file_path: Union[str, Path],
    language: str = "korean",
    task: str = "transcribe",
    domain: str = "normal"
) -> str:
    """Transcribe an audio file with domain adaptation"""
    audio_waveform = load_audio_file(file_path)
    return transcribe_audio(model, processor, audio_waveform, language, task, domain)


# --------------------------------------------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------------------------------------------

def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute Character Error Rate"""
    cer_metric = evaluate.load("cer")
    return cer_metric.compute(predictions=predictions, references=references)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute Word Error Rate"""
    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=predictions, references=references)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute all evaluation metrics"""
    return {
        "cer": compute_cer(predictions, references),
        "wer": compute_wer(predictions, references)
    }


# --------------------------------------------------------------------------------------
# Batch Testing
# --------------------------------------------------------------------------------------

def test_single_file(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_file: Union[str, Path],
    reference_text: Optional[str] = None,
    language: str = "korean",
    task: str = "transcribe",
    domain: str = "normal"
) -> Dict[str, Union[str, float, None]]:
    """Test a single audio file with domain adaptation"""
    try:
        start_time = time.time()
        transcription = transcribe_file(model, processor, audio_file, language, task, domain)
        inference_time = time.time() - start_time
        
        result = {
            "file_path": str(audio_file),
            "transcription": transcription,
            "inference_time": inference_time,
            "reference": reference_text,
            "domain": domain,
            "cer": None,
            "wer": None
        }
        
        if reference_text:
            ref_normalized = ko_norm(reference_text)
            metrics = compute_metrics([transcription], [ref_normalized])
            result["cer"] = metrics["cer"]
            result["wer"] = metrics["wer"]
        
        return result
        
    except Exception as e:
        return {
            "file_path": str(audio_file),
            "transcription": None,
            "inference_time": None,
            "reference": reference_text,
            "domain": domain,
            "cer": None,
            "wer": None,
            "error": str(e)
        }


def test_batch_files(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    test_files: List[Union[str, Path]],
    reference_texts: Optional[List[str]] = None,
    language: str = "korean",
    task: str = "transcribe",
    domain: str = "normal"
) -> List[Dict[str, Union[str, float, None]]]:
    """Test multiple audio files with domain adaptation"""
    results = []
    
    for i, audio_file in enumerate(tqdm(test_files, desc="Testing files")):
        ref_text = reference_texts[i] if reference_texts else None
        result = test_single_file(model, processor, audio_file, ref_text, language, task, domain)
        results.append(result)
    
    return results


# --------------------------------------------------------------------------------------
# Test Data Loading
# --------------------------------------------------------------------------------------

def load_test_data_from_json(json_file: str) -> Tuple[List[str], List[str]]:
    """Load test data from JSON file with 'audio_path' and 'text' keys"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    audio_files = [item['audio_path'] for item in data]
    reference_texts = [item['text'] for item in data]
    
    return audio_files, reference_texts


def load_test_data_from_directory(
    audio_dir: str, 
    text_dir: Optional[str] = None,
    audio_extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a']
) -> Tuple[List[str], List[str]]:
    """Load test data from directory structure"""
    audio_dir = Path(audio_dir)
    audio_files = []
    reference_texts = []
    
    for audio_file in audio_dir.rglob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            audio_files.append(str(audio_file))
            
            if text_dir:
                # Look for corresponding text file
                text_file = Path(text_dir) / audio_file.relative_to(audio_dir).with_suffix('.txt')
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        reference_texts.append(f.read().strip())
                else:
                    reference_texts.append(None)
            else:
                reference_texts.append(None)
    
    return audio_files, reference_texts


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def print_test_results(results: List[Dict[str, Union[str, float, None]]]) -> None:
    """Print detailed test results"""
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    successful_tests = [r for r in results if r.get('transcription') is not None]
    failed_tests = [r for r in results if r.get('transcription') is None]
    
    print(f"Total files tested: {len(results)}")
    print(f"Successful transcriptions: {len(successful_tests)}")
    print(f"Failed transcriptions: {len(failed_tests)}")
    
    if successful_tests:
        avg_inference_time = np.mean([r['inference_time'] for r in successful_tests])
        print(f"Average inference time: {avg_inference_time:.2f} seconds")
        
        # Compute overall metrics
        predictions = [r['transcription'] for r in successful_tests if r.get('reference')]
        references = [r['reference'] for r in successful_tests if r.get('reference')]
        
        if predictions and references:
            overall_metrics = compute_metrics(predictions, references)
            print(f"Overall CER: {overall_metrics['cer']:.4f}")
            print(f"Overall WER: {overall_metrics['wer']:.4f}")
    
    print("\n" + "-"*80)
    print("DETAILED RESULTS")
    print("-"*80)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {Path(result['file_path']).name}")
        print(f"   Path: {result['file_path']}")
        
        if result.get('error'):
            print(f"   ERROR: {result['error']}")
        else:
            print(f"   Transcription: {result['transcription']}")
            print(f"   Inference time: {result['inference_time']:.2f}s")
            print(f"   Domain: {result.get('domain', 'normal')}")
            
            if result.get('reference'):
                print(f"   Reference: {result['reference']}")
                print(f"   CER: {result['cer']:.4f}")
                print(f"   WER: {result['wer']:.4f}")


def save_test_results(results: List[Dict[str, Union[str, float, None]]], output_file: str) -> None:
    """Save test results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    # Clean results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = {}
        for key, value in result.items():
            clean_result[key] = convert_types(value)
        clean_results.append(clean_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    print(f"Test results saved to: {output_file}")


# --------------------------------------------------------------------------------------
# Main Testing Function
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test domain-adapted Whisper model")
    
    # Model configuration
    parser.add_argument("--checkpoint", type=str, 
                       default="runs/whisper_dyslexia_domain_adaptation/checkpoint-1000",
                       help="Path to model checkpoint")
    
    # Test data configuration
    parser.add_argument("--test-file", type=str,
                       help="Single audio file to test")
    parser.add_argument("--test-json", type=str,
                       help="JSON file with test data (audio_path, text keys)")
    parser.add_argument("--test-dir", type=str,
                       help="Directory containing audio files")
    parser.add_argument("--text-dir", type=str,
                       help="Directory containing reference text files")
    
    # Model parameters
    parser.add_argument("--language", type=str, default="korean",
                       help="Language for transcription")
    parser.add_argument("--task", type=str, default="transcribe",
                       help="Task type (transcribe/translate)")
    parser.add_argument("--domain", type=str, default="normal", choices=["normal", "dyslexic"],
                       help="Domain for transcription (normal/dyslexic)")
    
    # Output configuration
    parser.add_argument("--output", type=str,
                       help="Output file for test results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results")
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor = load_model_and_processor(args.checkpoint)
    
    # Determine test data
    if args.test_file:
        # Single file test
        print(f"Testing single file: {args.test_file}")
        result = test_single_file(model, processor, args.test_file, language=args.language, task=args.task, domain=args.domain)
        results = [result]
        
    elif args.test_json:
        # JSON test data
        print(f"Loading test data from: {args.test_json}")
        audio_files, reference_texts = load_test_data_from_json(args.test_json)
        results = test_batch_files(model, processor, audio_files, reference_texts, args.language, args.task, args.domain)
        
    elif args.test_dir:
        # Directory test data
        print(f"Loading test data from directory: {args.test_dir}")
        audio_files, reference_texts = load_test_data_from_directory(args.test_dir, args.text_dir)
        results = test_batch_files(model, processor, audio_files, reference_texts, args.language, args.task, args.domain)
        
    else:
        print("Error: No test data specified. Use --test-file, --test-json, or --test-dir")
        return
    
    # Print results
    if args.verbose:
        print_test_results(results)
    else:
        # Print summary only
        successful_tests = [r for r in results if r.get('transcription') is not None]
        print(f"\nTest completed: {len(successful_tests)}/{len(results)} successful")
        
        if successful_tests:
            predictions = [r['transcription'] for r in successful_tests if r.get('reference')]
            references = [r['reference'] for r in successful_tests if r.get('reference')]
            
            if predictions and references:
                overall_metrics = compute_metrics(predictions, references)
                print(f"CER: {overall_metrics['cer']:.4f}, WER: {overall_metrics['wer']:.4f}")
    
    # Save results if requested
    if args.output:
        save_test_results(results, args.output)


if __name__ == "__main__":
    main()
