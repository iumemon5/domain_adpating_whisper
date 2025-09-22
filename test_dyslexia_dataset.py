#!/usr/bin/env python3
"""
Specialized testing script for the dyslexia dataset with domain adaptation
- Tests the domain-adapted Whisper model on the dyslexia test dataset
- Supports webdataset format from the training pipeline
- Provides detailed analysis and comparison with reference texts
- Generates comprehensive evaluation reports with domain-specific metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Import our testing functions
from test_whisper_model import (
    load_model_and_processor,
    transcribe_file,
    compute_metrics,
    ko_norm
)
import webdataset as wds
import glob


def determine_domain_from_filename(filename: str) -> str:
    """
    Determine domain from filename pattern
    All test audio files are dyslexic
    """
    # All test files are dyslexic
    return "dyslexic"


def discover_test_files_directory(dataset_root: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Discover test files from directory structure (Audio/Script folders)
    Returns audio files, reference texts, and domain labels
    """
    audio_dir = Path(dataset_root) / "test" / "Audio"
    script_dir = Path(dataset_root) / "test" / "Script"
    
    if not audio_dir.exists() or not script_dir.exists():
        print(f"Audio or Script directory not found in {dataset_root}/test/")
        return [], [], []
    
    print(f"Scanning audio directory: {audio_dir}")
    print(f"Scanning script directory: {script_dir}")
    
    audio_files = []
    reference_texts = []
    domain_labels = []
    
    # Get all audio files recursively
    audio_paths = list(audio_dir.rglob("*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    for audio_path in tqdm(audio_paths, desc="Matching audio-text pairs"):
        # Find corresponding text file
        relative_path = audio_path.relative_to(audio_dir)
        text_path = script_dir / relative_path.with_suffix('.txt')
        
        if text_path.exists():
            # Read reference text
            with open(text_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
            
            # Determine domain from filename
            domain = determine_domain_from_filename(audio_path.name)
            
            audio_files.append(str(audio_path))
            reference_texts.append(reference_text)
            domain_labels.append(domain)
        else:
            print(f"Warning: No reference text found for {audio_path}")
    
    print(f"Successfully matched {len(audio_files)} audio-text pairs")
    return audio_files, reference_texts, domain_labels


def discover_test_files(dataset_root: str) -> Tuple[List[str], List[str]]:
    """
    Discover all test audio files and their corresponding reference texts
    from the dyslexia dataset structure
    """
    audio_dir = Path(dataset_root) / "test" / "Audio"
    script_dir = Path(dataset_root) / "test" / "Script"
    
    audio_files = []
    reference_texts = []
    
    print(f"Scanning audio directory: {audio_dir}")
    print(f"Scanning script directory: {script_dir}")
    
    # Get all audio files
    audio_paths = list(audio_dir.rglob("*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    for audio_path in tqdm(audio_paths, desc="Matching audio-text pairs"):
        # Find corresponding text file
        relative_path = audio_path.relative_to(audio_dir)
        text_path = script_dir / relative_path.with_suffix('.txt')
        
        if text_path.exists():
            # Read reference text
            with open(text_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
            
            audio_files.append(str(audio_path))
            reference_texts.append(reference_text)
        else:
            print(f"Warning: No reference text found for {audio_path}")
    
    print(f"Successfully matched {len(audio_files)} audio-text pairs")
    return audio_files, reference_texts


def analyze_dataset_statistics(audio_files: List[str], reference_texts: List[str]) -> Dict:
    """Analyze dataset statistics"""
    stats = {
        "total_files": len(audio_files),
        "total_characters": 0,
        "total_words": 0,
        "avg_chars_per_file": 0,
        "avg_words_per_file": 0,
        "char_lengths": [],
        "word_lengths": [],
        "speaker_ids": set(),
        "dialect_codes": set()
    }
    
    for i, (audio_file, ref_text) in enumerate(zip(audio_files, reference_texts)):
        # Extract speaker info from filename
        filename = Path(audio_file).name
        parts = filename.split('_')
        if len(parts) >= 1:
            speaker_id = parts[0]
            stats["speaker_ids"].add(speaker_id)
            
            # Extract dialect code (last part before file extension)
            if len(parts) >= 2:
                dialect_code = parts[-1].split('.')[0]
                stats["dialect_codes"].add(dialect_code)
        
        # Text statistics
        char_count = len(ref_text)
        word_count = len(ref_text.split())
        
        stats["total_characters"] += char_count
        stats["total_words"] += word_count
        stats["char_lengths"].append(char_count)
        stats["word_lengths"].append(word_count)
    
    # Calculate averages
    stats["avg_chars_per_file"] = stats["total_characters"] / len(audio_files) if audio_files else 0
    stats["avg_words_per_file"] = stats["total_words"] / len(audio_files) if audio_files else 0
    
    # Convert sets to lists for JSON serialization
    stats["speaker_ids"] = list(stats["speaker_ids"])
    stats["dialect_codes"] = list(stats["dialect_codes"])
    
    return stats


def test_dyslexia_dataset(
    checkpoint_path: str,
    dataset_root: str,
    output_file: Optional[str] = None,
    max_files: Optional[int] = None,
    verbose: bool = False,
    use_webdataset: bool = True
) -> Dict:
    """Test the domain-adapted model on the dyslexia dataset"""
    
    print("="*80)
    print("DYSLEXIA DATASET TESTING (DOMAIN ADAPTATION)")
    print("="*80)
    print("Note: All test audio files are dyslexic")
    
    # Discover test files
    if use_webdataset:
        audio_files, reference_texts, domain_labels = discover_test_files_webdataset(dataset_root)
    else:
        audio_files, reference_texts, domain_labels = discover_test_files_directory(dataset_root)
    
    if not audio_files:
        print("No test files found!")
        return {"error": "No test files found"}
    
    if max_files:
        audio_files = audio_files[:max_files]
        reference_texts = reference_texts[:max_files]
        domain_labels = domain_labels[:max_files]
        print(f"Limited to first {max_files} files for testing")
    
    # Analyze dataset statistics
    print("\nAnalyzing dataset statistics...")
    dataset_stats = analyze_dataset_statistics(audio_files, reference_texts)
    
    # Add domain statistics
    domain_counts = {}
    for domain in domain_labels:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    dataset_stats['domain_counts'] = domain_counts
    
    print(f"Total test files: {dataset_stats['total_files']}")
    print(f"Domain distribution: {domain_counts}")
    print(f"Average characters per file: {dataset_stats['avg_chars_per_file']:.1f}")
    print(f"Average words per file: {dataset_stats['avg_words_per_file']:.1f}")
    
    # Load model
    print(f"\nLoading domain-adapted model from: {checkpoint_path}")
    model, processor = load_model_and_processor(checkpoint_path)
    
    # Test all files
    print(f"\nTesting {len(audio_files)} files...")
    results = []
    predictions = []
    references = []
    
    # Separate results by domain
    domain_results = {"normal": [], "dyslexic": []}
    
    for i, (audio_file, ref_text, domain) in enumerate(tqdm(zip(audio_files, reference_texts, domain_labels), 
                                                           total=len(audio_files), 
                                                           desc="Testing files")):
        try:
            # Transcribe with domain-specific setting
            transcription = transcribe_file(model, processor, audio_file, domain=domain)
            
            # Normalize texts
            transcription_norm = ko_norm(transcription)
            reference_norm = ko_norm(ref_text)
            
            # Compute metrics
            metrics = compute_metrics([transcription_norm], [reference_norm])
            
            result = {
                "file_index": i,
                "audio_file": audio_file,
                "reference_text": ref_text,
                "transcription": transcription,
                "domain": domain,
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "char_count": len(ref_text),
                "word_count": len(ref_text.split())
            }
            
            results.append(result)
            domain_results[domain].append(result)
            predictions.append(transcription_norm)
            references.append(reference_norm)
            
            if verbose and i < 10:  # Show first 10 examples
                print(f"\nFile {i+1}: {Path(audio_file).name}")
                print(f"Domain: {domain}")
                print(f"Reference: {ref_text}")
                print(f"Transcription: {transcription}")
                print(f"CER: {metrics['cer']:.4f}, WER: {metrics['wer']:.4f}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results.append({
                "file_index": i,
                "audio_file": audio_file,
                "reference_text": ref_text,
                "transcription": None,
                "domain": domain,
                "cer": None,
                "wer": None,
                "error": str(e)
            })
    
    # Compute overall metrics
    successful_results = [r for r in results if r.get('transcription') is not None]
    if successful_results:
        overall_metrics = compute_metrics(predictions, references)
        
        # Calculate additional statistics
        cers = [r['cer'] for r in successful_results if r['cer'] is not None]
        wers = [r['wer'] for r in successful_results if r['wer'] is not None]
        
        # Calculate per-domain metrics
        domain_metrics = {}
        for domain in ["normal", "dyslexic"]:
            domain_successful = [r for r in domain_results[domain] if r.get('transcription') is not None]
            if domain_successful:
                domain_predictions = [ko_norm(r['transcription']) for r in domain_successful]
                domain_references = [ko_norm(r['reference_text']) for r in domain_successful]
                domain_metrics[domain] = compute_metrics(domain_predictions, domain_references)
        
        test_summary = {
            "dataset_stats": dataset_stats,
            "test_results": results,
            "overall_metrics": overall_metrics,
            "domain_metrics": domain_metrics,
            "cer_stats": {
                "mean": np.mean(cers),
                "std": np.std(cers),
                "min": np.min(cers),
                "max": np.max(cers),
                "median": np.median(cers)
            },
            "wer_stats": {
                "mean": np.mean(wers),
                "std": np.std(wers),
                "min": np.min(wers),
                "max": np.max(wers),
                "median": np.median(wers)
            },
            "successful_tests": len(successful_results),
            "failed_tests": len(results) - len(successful_results)
        }
    else:
        test_summary = {
            "dataset_stats": dataset_stats,
            "test_results": results,
            "error": "No successful tests"
        }
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if successful_results:
        print(f"Successful tests: {len(successful_results)}/{len(results)}")
        print(f"Overall CER: {overall_metrics['cer']:.4f}")
        print(f"Overall WER: {overall_metrics['wer']:.4f}")
        print(f"CER - Mean: {np.mean(cers):.4f}, Std: {np.std(cers):.4f}")
        print(f"WER - Mean: {np.mean(wers):.4f}, Std: {np.std(wers):.4f}")
        
        # Print per-domain metrics
        print("\nPer-domain metrics:")
        for domain, metrics in domain_metrics.items():
            print(f"  {domain.capitalize()} domain: CER={metrics['cer']:.4f}, WER={metrics['wer']:.4f}")
    else:
        print("No successful tests completed")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return test_summary


def main():
    parser = argparse.ArgumentParser(description="Test domain-adapted Whisper model on dyslexia dataset")
    
    parser.add_argument("--checkpoint", type=str, 
                       default="runs/whisper_dyslexia_domain_adaptation/checkpoint-16650",
                       help="Path to model checkpoint")
    
    parser.add_argument("--dataset-root", type=str,
                       default="/home/braindeck/ssd/irfan/dataset/dyslexia_dataset_webdataset",
                       help="Root directory of dyslexia dataset") 
    
    parser.add_argument("--output", type=str,
                       help="Output file for test results (JSON)")
    
    parser.add_argument("--max-files", type=int,
                       help="Maximum number of files to test (for quick testing)")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results for first 10 files")
    
    parser.add_argument("--use-webdataset", action="store_true", default=False,
                       help="Use webdataset format (default: False)")
    
    parser.add_argument("--use-traditional", dest="use_webdataset", action="store_false",
                       help="Use traditional directory format instead of webdataset")
    
    args = parser.parse_args()
    
    # Run the test
    test_dyslexia_dataset(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_file=args.output,
        max_files=args.max_files,
        verbose=args.verbose,
        use_webdataset=args.use_webdataset
    )


if __name__ == "__main__":
    main()
