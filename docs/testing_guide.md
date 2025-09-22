# Testing Guide

This guide provides comprehensive instructions for testing and evaluating domain-adapted Whisper models.

## Overview

The testing framework supports multiple evaluation scenarios:
- Single file testing
- Batch processing
- Dataset-wide evaluation
- Domain-specific performance analysis

## Prerequisites

### Model Requirements

- Trained domain-adapted model checkpoint
- Compatible Whisper processor
- Sufficient GPU memory (8GB+ recommended)

### Test Data

- Audio files in supported formats (WAV, MP3, FLAC, M4A)
- Reference transcriptions (optional but recommended)
- Domain labels (normal/dyslexic)

## Testing Scripts

### 1. General Testing (`test_whisper_model.py`)

For testing individual files or custom datasets.

#### Single File Testing

```bash
# Test with normal domain
python test_whisper_model.py \
    --test-file audio.wav \
    --domain normal \
    --verbose

# Test with dyslexic domain
python test_whisper_model.py \
    --test-file audio.wav \
    --domain dyslexic \
    --verbose
```

#### Batch Testing with JSON

Create test data file (`test_data.json`):

```json
[
  {
    "audio_path": "audio1.wav",
    "text": "안녕하세요, 이것은 테스트입니다.",
    "domain": "normal"
  },
  {
    "audio_path": "audio2.wav",
    "text": "한국어 음성 인식 모델을 테스트하고 있습니다.",
    "domain": "dyslexic"
  }
]
```

Run batch test:

```bash
python test_whisper_model.py \
    --test-json test_data.json \
    --verbose \
    --output results.json
```

#### Directory Testing

```bash
python test_whisper_model.py \
    --test-dir audio_files/ \
    --text-dir reference_texts/ \
    --domain normal \
    --verbose
```

### 2. Dataset Testing (`test_dyslexia_dataset.py`)

For testing on the full dyslexia dataset.

```bash
python test_dyslexia_dataset.py \
    --dataset-root /path/to/dataset \
    --checkpoint runs/whisper_dyslexia_domain_adaptation/checkpoint-1000 \
    --verbose \
    --output evaluation.json
```

## Configuration Options

### Model Configuration

- `--checkpoint`: Path to model checkpoint
- `--language`: Language for transcription (default: korean)
- `--task`: Task type - transcribe or translate (default: transcribe)

### Domain Settings

- `--domain`: Domain for transcription
  - `normal`: For typical speech patterns
  - `dyslexic`: For dyslexic speech patterns

### Output Configuration

- `--output`: Output file for results (JSON format)
- `--verbose`: Print detailed results
- `--max-files`: Limit number of files for quick testing

## Evaluation Metrics

### Character Error Rate (CER)

Measures character-level accuracy:

```
CER = (Substitutions + Insertions + Deletions) / Total Characters
```

### Word Error Rate (WER)

Measures word-level accuracy:

```
WER = (Substitutions + Insertions + Deletions) / Total Words
```

### Domain-Specific Metrics

Separate evaluation for each domain:
- Normal domain performance
- Dyslexic domain performance
- Cross-domain comparison

## Test Data Formats

### Supported Audio Formats

- **WAV**: Recommended format
- **MP3**: Common compressed format
- **FLAC**: Lossless compression
- **M4A**: Apple audio format

### Directory Structure

```
test_data/
├── audio_files/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── subfolder/
│       └── sample3.wav
└── reference_texts/
    ├── sample1.txt
    ├── sample2.txt
    └── subfolder/
        └── sample3.txt
```

### JSON Format

```json
[
  {
    "audio_path": "path/to/audio.wav",
    "text": "Reference transcription",
    "domain": "normal"
  }
]
```

## Performance Analysis

### Single File Analysis

```bash
# Test same file with both domains
python test_whisper_model.py --test-file audio.wav --domain normal --verbose
python test_whisper_model.py --test-file audio.wav --domain dyslexic --verbose
```

Compare results:
- Transcription accuracy
- CER/WER metrics
- Inference time
- Domain-specific improvements

### Batch Analysis

```bash
# Test batch with normal domain
python test_whisper_model.py --test-json test_data.json --domain normal --output normal_results.json

# Test batch with dyslexic domain
python test_whisper_model.py --test-json test_data.json --domain dyslexic --output dyslexic_results.json
```

### Statistical Analysis

The testing scripts provide:

- **Overall metrics**: Average CER/WER across all files
- **Per-file metrics**: Individual file performance
- **Domain comparison**: Normal vs. dyslexic performance
- **Error analysis**: Common error patterns
- **Timing statistics**: Inference speed analysis

## Advanced Testing

### Custom Evaluation

```python
from test_whisper_model import load_model_and_processor, test_single_file

# Load model
model, processor = load_model_and_processor("checkpoint_path")

# Test custom audio
result = test_single_file(
    model=model,
    processor=processor,
    audio_file="custom_audio.wav",
    reference_text="Expected transcription",
    domain="dyslexic"
)

print(f"CER: {result['cer']:.4f}")
print(f"WER: {result['wer']:.4f}")
```

### Error Analysis

```python
# Analyze errors in detail
def analyze_errors(predictions, references):
    from evaluate import load
    
    cer_metric = load("cer")
    wer_metric = load("wer")
    
    # Compute detailed metrics
    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Analyze error patterns
    for pred, ref in zip(predictions, references):
        if pred != ref:
            print(f"Prediction: {pred}")
            print(f"Reference: {ref}")
            print("---")
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

```bash
# Check checkpoint path
ls -la runs/whisper_dyslexia_domain_adaptation/

# Verify model files exist
ls -la checkpoint-1000/
```

#### Audio Processing Errors

```bash
# Check audio file format
file audio.wav

# Verify audio can be loaded
python -c "import soundfile as sf; print(sf.info('audio.wav'))"
```

#### Memory Issues

```bash
# Reduce batch size for testing
python test_whisper_model.py --test-json test_data.json --max-files 10
```

### Debugging Tips

1. **Start with single file**: Test one file first
2. **Check audio format**: Ensure supported format
3. **Verify model loading**: Check checkpoint path
4. **Monitor GPU memory**: Use `nvidia-smi`
5. **Enable verbose output**: Use `--verbose` flag

## Best Practices

### Test Data Preparation

- Use high-quality audio recordings
- Ensure accurate reference transcriptions
- Include diverse speakers and contexts
- Balance normal and dyslexic samples

### Evaluation Strategy

- Test both domains for comparison
- Use representative test sets
- Include edge cases and difficult samples
- Monitor inference time and resource usage

### Result Interpretation

- Compare CER and WER metrics
- Analyze domain-specific improvements
- Consider practical significance
- Document performance baselines

## Expected Results

### Performance Benchmarks

Typical performance on dyslexic speech:

- **Base Whisper**: CER ~0.15-0.25
- **Domain-Adapted**: CER ~0.10-0.20
- **Improvement**: 15-25% reduction in CER

### Domain Comparison

- **Normal domain**: Better for typical speech
- **Dyslexic domain**: Better for dyslexic speech patterns
- **Cross-domain**: May show different strengths

## Integration

### Application Integration

```python
# Load model for production use
model, processor = load_model_and_processor("production_checkpoint")

def transcribe_audio(audio_file, domain="normal"):
    result = test_single_file(
        model, processor, audio_file, domain=domain
    )
    return result["transcription"]
```

### API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model, processor = load_model_and_processor("checkpoint")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audio"]
    domain = request.form.get("domain", "normal")
    
    result = test_single_file(model, processor, audio_file, domain=domain)
    return jsonify(result)
```

For training instructions, see [Training Guide](training_guide.md).
