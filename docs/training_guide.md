# Training Guide

This guide provides comprehensive instructions for training domain-adapted Whisper models for dyslexic speech recognition.

## Overview

The training process involves fine-tuning OpenAI's Whisper model with domain-specific tokens and selective layer unfreezing to improve recognition accuracy for dyslexic speech patterns.

## Prerequisites

### Hardware Requirements

- **GPU**: CUDA-compatible GPU with at least 16GB VRAM (24GB+ recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space for datasets and checkpoints

### Software Requirements

- Python 3.8+
- CUDA 11.8+ (if using GPU)
- Required Python packages (see `requirements.txt`)

## Dataset Preparation

### Dataset Structure

Your dataset should be organized in WebDataset format:

```
dyslexia_dataset_webdataset/
├── train/
│   ├── shard_000000.tar
│   ├── shard_000001.tar
│   ├── ...
│   └── metadata.json
└── val_small/
    ├── shard_000000.tar
    ├── shard_000001.tar
    ├── ...
    └── metadata.json
```

### Metadata Format

Each metadata.json should contain:

```json
{
  "total_samples": 10000,
  "shards": [
    {
      "filename": "shard_000000.tar",
      "samples": 1000
    }
  ]
}
```

### Audio Format

- **Sample Rate**: 16kHz (will be resampled if different)
- **Format**: WAV, MP3, FLAC, or M4A
- **Channels**: Mono or stereo (will be converted to mono)
- **Duration**: Variable length supported

### Text Format

- **Language**: Korean text
- **Encoding**: UTF-8
- **Normalization**: Automatic Korean text normalization applied

## Training Configuration

### Basic Training

```bash
python whisper_dyslexia_cross_domain.py \
    --dataset-root /path/to/dataset \
    --run-name my_experiment \
    --num-train-epochs 100
```

### Advanced Configuration

```bash
python whisper_dyslexia_cross_domain.py \
    --dataset-root /path/to/dataset \
    --base-model openai/whisper-large-v3 \
    --run-name advanced_experiment \
    --learning-rate 1e-4 \
    --num-train-epochs 200 \
    --per-device-train-batch-size 8 \
    --grad-accum 4 \
    --warmup-steps 2000 \
    --eval-steps 1000 \
    --save-steps 1000 \
    --bf16 \
    --login-wandb \
    --login-hf
```

## Key Parameters

### Dataset Configuration

- `--dataset-root`: Path to WebDataset directory
- `--target-sr`: Target sample rate (default: 16000)
- `--samples-per-shard`: Fallback estimate for samples per shard

### Model Configuration

- `--base-model`: Base Whisper model to fine-tune
  - `openai/whisper-large-v3` (recommended)
  - `openai/whisper-medium`
  - `openai/whisper-small`

### Training Hyperparameters

- `--learning-rate`: Learning rate (default: 1e-4)
- `--num-train-epochs`: Number of training epochs
- `--per-device-train-batch-size`: Batch size per device
- `--grad-accum`: Gradient accumulation steps
- `--warmup-steps`: Number of warmup steps
- `--lr-scheduler-type`: Learning rate scheduler (default: cosine)

### Evaluation and Checkpointing

- `--eval-steps`: Steps between evaluations
- `--save-steps`: Steps between model saves
- `--save-total-limit`: Maximum checkpoints to keep

### Performance Optimization

- `--bf16`: Use bfloat16 mixed precision
- `--num-workers`: Number of dataloader workers
- `--persistent-workers`: Use persistent dataloader workers

## Domain Adaptation Details

### Domain Tokens

The model uses special tokens to guide transcription:

- `<|domain:normal|>`: For typical speech patterns
- `<|domain:dyslexic|>`: For dyslexic speech patterns

### Layer Unfreezing Strategy

The training script implements selective fine-tuning:

1. **Freeze all parameters** initially
2. **Unfreeze last 4 encoder blocks** for acoustic adaptation
3. **Unfreeze last 4 decoder blocks** (attention + MLP layers only)

This approach:
- Preserves general speech recognition capabilities
- Allows adaptation to dyslexic speech patterns
- Reduces overfitting risk
- Maintains computational efficiency

### Data Collator

The `DataCollatorSpeechSeq2SeqDomain` automatically:

1. Extracts domain information from data
2. Prepends appropriate domain token
3. Handles padding and attention masks
4. Prepares labels for training

## Monitoring Training

### Weights & Biases Integration

Enable W&B logging:

```bash
export WANDB_API_KEY="your_api_key"
python whisper_dyslexia_cross_domain.py --login-wandb
```

W&B will track:
- Training and validation loss
- Character Error Rate (CER)
- Learning rate schedule
- GPU utilization
- Model parameters

### Hugging Face Hub Integration

Enable model pushing:

```bash
export HF_TOKEN="your_token"
python whisper_dyslexia_cross_domain.py --login-hf
```

Models are automatically pushed to:
- `braindeck/{run_name}` (private by default)
- Includes model weights, config, and training logs

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```bash
# Reduce batch size
--per-device-train-batch-size 4

# Increase gradient accumulation
--grad-accum 8

# Use gradient checkpointing
--gradient_checkpointing
```

#### Slow Training

```bash
# Increase number of workers
--num-workers 24

# Enable persistent workers
--persistent-workers

# Use mixed precision
--bf16
```

#### Poor Convergence

```bash
# Adjust learning rate
--learning-rate 5e-5

# Increase warmup steps
--warmup-steps 2000

# Check data quality and preprocessing
```

### Debugging Tips

1. **Start with small dataset**: Test with subset first
2. **Monitor loss curves**: Look for proper convergence
3. **Check data loading**: Ensure WebDataset is loading correctly
4. **Validate preprocessing**: Check audio and text preprocessing
5. **Use verbose logging**: Enable detailed output

## Best Practices

### Data Quality

- Ensure high-quality audio recordings
- Verify text transcriptions are accurate
- Balance normal vs. dyslexic samples
- Include diverse speakers and contexts

### Training Strategy

- Start with pre-trained Whisper models
- Use appropriate learning rates (1e-4 to 1e-5)
- Monitor validation metrics closely
- Save checkpoints regularly
- Use early stopping if overfitting

### Hyperparameter Tuning

- Learning rate: 1e-5 to 1e-4
- Batch size: 4-16 (depending on GPU memory)
- Warmup steps: 500-2000
- Epochs: 50-200 (depending on dataset size)

## Expected Results

### Performance Metrics

Typical improvements with domain adaptation:

- **CER Reduction**: 15-25% on dyslexic speech
- **WER Improvement**: Better word-level accuracy
- **Domain-Specific Gains**: Significant improvement on dyslexic patterns

### Training Time

Approximate training times (RTX 4090, 24GB VRAM):

- Small dataset (10K samples): 2-4 hours
- Medium dataset (100K samples): 1-2 days
- Large dataset (1M+ samples): 1-2 weeks

## Next Steps

After training:

1. **Evaluate model**: Use testing scripts to assess performance
2. **Compare domains**: Test both normal and dyslexic domains
3. **Deploy model**: Integrate into applications
4. **Iterate**: Fine-tune based on evaluation results

For testing instructions, see [Testing Guide](testing_guide.md).
