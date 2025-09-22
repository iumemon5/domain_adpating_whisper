# Domain-Adapted Whisper Model Testing Script

This directory contains a comprehensive testing script for the domain-adapted Whisper model trained using `whisper_dyslexia_cross_domain.py`.

## Files

- `test_whisper_model.py` - Main testing script with domain adaptation support
- `test_dyslexia_dataset.py` - Specialized testing for dyslexia dataset with webdataset support
- `test_examples.py` - Usage examples and documentation
- `runs/whisper_dyslexia_domain_adaptation/` - Domain-adapted model checkpoints

## Quick Start

### Test a single audio file with domain adaptation:
```bash
# Test with normal domain
python test_whisper_model.py --test-file /path/to/audio.wav --domain normal --verbose

# Test with dyslexic domain
python test_whisper_model.py --test-file /path/to/audio.wav --domain dyslexic --verbose
```

### Test multiple files with reference texts:
```bash
python test_whisper_model.py --test-json test_data.json --verbose --output results.json
```

### Test directory of audio files:
```bash
python test_whisper_model.py --test-dir audio_files/ --text-dir reference_texts/ --domain normal --verbose
```

### Test dyslexia dataset with webdataset format:
```bash
python test_dyslexia_dataset.py --dataset-root dyslexia_dataset_webdataset --verbose --output results.json
```

## Features

- **Domain adaptation**: Test with normal or dyslexic domain settings
- **Single file testing**: Test individual audio files with domain-specific transcription
- **Batch testing**: Test multiple files with progress bar
- **Webdataset support**: Test directly on webdataset format from training pipeline
- **Evaluation metrics**: Computes CER (Character Error Rate) and WER (Word Error Rate)
- **Per-domain metrics**: Separate evaluation metrics for normal and dyslexic domains
- **Multiple input formats**: Supports JSON configuration, directory structure, or single files
- **Detailed reporting**: Verbose output with timing and error analysis
- **JSON export**: Save results for further analysis
- **GPU acceleration**: Automatic GPU usage when available

## Model Configuration

The script automatically loads the best checkpoint from `runs/whisper_dyslexia_domain_adaptation/` and uses the same preprocessing pipeline as the training script. The model includes domain-specific tokens (`<|domain:normal|>` and `<|domain:dyslexic|>`) for improved transcription accuracy.

## Supported Audio Formats

- WAV
- MP3  
- FLAC
- M4A

## Requirements

- torch
- transformers
- soundfile
- torchaudio
- evaluate
- tqdm
- numpy

## Example JSON Test Data Format

```json
[
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
```

## Command Line Options

### test_whisper_model.py
- `--checkpoint`: Path to model checkpoint (default: runs/whisper_dyslexia_domain_adaptation/checkpoint-1000)
- `--test-file`: Single audio file to test
- `--test-json`: JSON file with test data
- `--test-dir`: Directory containing audio files
- `--text-dir`: Directory containing reference text files
- `--domain`: Domain for transcription (normal/dyslexic, default: normal)
- `--language`: Language for transcription (default: korean)
- `--task`: Task type (transcribe/translate, default: transcribe)
- `--output`: Output file for test results (JSON)
- `--verbose`: Print detailed results

### test_dyslexia_dataset.py
- `--checkpoint`: Path to model checkpoint (default: runs/whisper_dyslexia_domain_adaptation/checkpoint-1000)
- `--dataset-root`: Root directory of dyslexia dataset (default: dyslexia_dataset_webdataset)
- `--use-webdataset`: Use webdataset format (default: True)
- `--use-traditional`: Use traditional directory format instead of webdataset
- `--output`: Output file for test results (JSON)
- `--max-files`: Maximum number of files to test (for quick testing)
- `--verbose`: Print detailed results for first 10 files

## Domain Adaptation Features

The domain-adapted Whisper model includes special tokens for improved transcription accuracy:

- **Normal Domain**: Uses `<|domain:normal|>` token for regular speech patterns
- **Dyslexic Domain**: Uses `<|domain:dyslexic|>` token for dyslexic speech patterns

### When to Use Each Domain

- **Normal Domain**: Use for typical speech patterns, clear pronunciation, and standard Korean speech
- **Dyslexic Domain**: Use for speech with dyslexic characteristics, such as:
  - Phonological processing difficulties
  - Word substitution errors
  - Pronunciation variations
  - Speech disfluencies

### Testing Both Domains

To compare performance across domains:

```bash
# Test same audio with both domains
python test_whisper_model.py --test-file audio.wav --domain normal --verbose
python test_whisper_model.py --test-file audio.wav --domain dyslexic --verbose
```

For more examples, run:
```bash
python test_examples.py
```
