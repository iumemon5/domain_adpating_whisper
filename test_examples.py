#!/usr/bin/env python3
"""
Example usage of the domain-adapted Whisper model testing script
"""

import json
import os
from pathlib import Path

# Example 1: Test a single audio file
def test_single_file_example():
    """Example of testing a single audio file with domain adaptation"""
    print("Example 1: Testing a single audio file")
    print("Command:")
    print("python test_whisper_model.py --test-file /path/to/audio.wav --domain normal --verbose")
    print("python test_whisper_model.py --test-file /path/to/audio.wav --domain dyslexic --verbose")
    print()

# Example 2: Test multiple files with JSON configuration
def test_json_example():
    """Example of testing with JSON configuration"""
    print("Example 2: Testing with JSON configuration")
    
    # Create example JSON file
    example_data = [
        {
            "audio_path": "/path/to/audio1.wav",
            "text": "안녕하세요, 이것은 테스트입니다.",
            "domain": "normal"
        },
        {
            "audio_path": "/path/to/audio2.wav", 
            "text": "한국어 음성 인식 모델을 테스트하고 있습니다.",
            "domain": "dyslexic"
        }
    ]
    
    with open("example_test_data.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    print("Created example_test_data.json")
    print("Command:")
    print("python test_whisper_model.py --test-json example_test_data.json --verbose --output results.json")
    print()

# Example 3: Test directory with reference texts
def test_directory_example():
    """Example of testing directory structure"""
    print("Example 3: Testing directory structure")
    print("Directory structure:")
    print("test_audio/")
    print("├── audio1.wav")
    print("├── audio2.wav")
    print("└── subfolder/")
    print("    └── audio3.wav")
    print()
    print("test_texts/")
    print("├── audio1.txt")
    print("├── audio2.txt")
    print("└── subfolder/")
    print("    └── audio3.txt")
    print()
    print("Command:")
    print("python test_whisper_model.py --test-dir test_audio --text-dir test_texts --verbose")
    print()

# Example 4: Domain-specific testing
def test_domain_specific_example():
    """Example of testing with specific domain settings"""
    print("Example 4: Domain-specific testing")
    print("Test normal domain:")
    print("python test_whisper_model.py --test-dir test_audio --domain normal --verbose")
    print()
    print("Test dyslexic domain:")
    print("python test_whisper_model.py --test-dir test_audio --domain dyslexic --verbose")
    print()

# Example 5: Quick test without reference texts
def test_quick_example():
    """Example of quick testing without reference texts"""
    print("Example 5: Quick testing (no reference texts)")
    print("Command:")
    print("python test_whisper_model.py --test-dir test_audio --domain normal --output results.json")
    print()

def main():
    print("Domain-Adapted Whisper Model Testing Script - Usage Examples")
    print("=" * 60)
    print()
    
    test_single_file_example()
    test_json_example()
    test_directory_example()
    test_domain_specific_example()
    test_quick_example()
    
    print("Additional Options:")
    print("--checkpoint: Specify different checkpoint (default: runs/whisper_dyslexia_domain_adaptation/checkpoint-1000)")
    print("--domain: Domain for transcription - normal or dyslexic (default: normal)")
    print("--language: Language for transcription (default: korean)")
    print("--task: Task type - transcribe or translate (default: transcribe)")
    print("--verbose: Print detailed results")
    print("--output: Save results to JSON file")
    print()
    
    print("Domain Adaptation Features:")
    print("- Use --domain normal for regular speech")
    print("- Use --domain dyslexic for dyslexic speech patterns")
    print("- Model automatically applies appropriate domain tokens")
    print()
    
    print("Note: Make sure your audio files are in supported formats:")
    print("WAV, MP3, FLAC, M4A")

if __name__ == "__main__":
    main()
