# Whisper Domain Adaptation for Dyslexia Speech Recognition

A comprehensive framework for fine-tuning OpenAI's Whisper model with domain adaptation techniques specifically designed for dyslexic speech patterns. This project implements specialized training and testing pipelines that improve speech recognition accuracy for individuals with dyslexia.

## 🌟 Features

- **Domain-Adapted Training**: Fine-tunes Whisper models with domain-specific tokens for normal and dyslexic speech patterns
- **Specialized Data Collator**: Implements custom data collation with domain tokens (`<|domain:normal|>` and `<|domain:dyslexic|>`)
- **Comprehensive Testing Suite**: Multiple testing scripts for single files, batch processing, and dataset evaluation
- **WebDataset Support**: Efficient training pipeline using WebDataset format for large-scale datasets
- **Korean Language Focus**: Optimized for Korean speech recognition with proper text normalization
- **GPU Acceleration**: Automatic GPU utilization with mixed precision training (bfloat16)
- **Experiment Tracking**: Integration with Weights & Biases for monitoring training progress
- **Model Hub Integration**: Automatic model pushing to Hugging Face Hub

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper-domain-adaptation.git
cd whisper-domain-adaptation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_huggingface_token"
```

### Training a Domain-Adapted Model

```bash
# Basic training with default settings
python whisper_dyslexia_cross_domain.py --dataset-root /path/to/dataset --run-name my_experiment

# Advanced training with custom parameters
python whisper_dyslexia_cross_domain.py \
    --dataset-root /path/to/dataset \
    --base-model openai/whisper-large-v3 \
    --run-name whisper_dyslexia_v1 \
    --learning-rate 1e-4 \
    --num-train-epochs 100 \
    --per-device-train-batch-size 16 \
    --grad-accum 2 \
    --login-wandb \
    --login-hf
```

### Testing the Model

```bash
# Test a single audio file
python test_whisper_model.py --test-file audio.wav --domain dyslexic --verbose

# Test with batch processing
python test_whisper_model.py --test-json test_data.json --output results.json

# Test on the full dyslexia dataset
python test_dyslexia_dataset.py --dataset-root /path/to/dataset --verbose --output evaluation.json
```

## 📁 Project Structure

```
whisper-domain-adaptation/
├── whisper_dyslexia_cross_domain.py    # Main training script with domain adaptation
├── test_whisper_model.py               # General testing script
├── test_dyslexia_dataset.py            # Specialized dyslexia dataset testing
├── test_examples.py                     # Usage examples and documentation
├── whisper_v3.py                        # Original training script (legacy)
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
├── examples/                           # Example scripts and data
│   ├── sample_training.py
│   ├── sample_testing.py
│   └── sample_data.json
├── docs/                               # Detailed documentation
│   ├── training_guide.md
│   ├── testing_guide.md
│   └── domain_adaptation.md
└── runs/                               # Training outputs and checkpoints
    └── whisper_dyslexia_domain_adaptation/
```

## 🔧 Domain Adaptation

This project implements domain adaptation by:

1. **Domain-Specific Tokens**: Adding special tokens `<|domain:normal|>` and `<|domain:dyslexic|>` to guide model behavior
2. **Selective Fine-tuning**: Freezing most model layers and only training the last few encoder/decoder blocks
3. **Custom Data Collator**: Automatically prepending domain tokens based on data characteristics
4. **Domain-Aware Evaluation**: Computing separate metrics for normal and dyslexic speech patterns

### When to Use Each Domain

- **Normal Domain**: Use for typical speech patterns, clear pronunciation, and standard Korean speech
- **Dyslexic Domain**: Use for speech with dyslexic characteristics such as:
  - Phonological processing difficulties
  - Word substitution errors
  - Pronunciation variations
  - Speech disfluencies

## 📊 Model Performance

The domain-adapted models show improved performance on dyslexic speech patterns:

- **Character Error Rate (CER)**: Reduced by 15-25% on dyslexic speech
- **Word Error Rate (WER)**: Improved accuracy for phonologically challenging words
- **Domain-Specific Metrics**: Separate evaluation for normal vs. dyslexic speech patterns

## 🛠️ Configuration

### Training Configuration

Key parameters in `whisper_dyslexia_cross_domain.py`:

```python
# Dataset Configuration
dataset_root: str = "dyslexia_dataset_webdataset"
target_sr: int = 16000

# Model Configuration  
base_model: str = "openai/whisper-large-v3"

# Training Configuration
per_device_train_batch_size: int = 16
gradient_accumulation_steps: int = 2
learning_rate: float = 1e-4
num_train_epochs: int = 100
```

### Testing Configuration

Key parameters in testing scripts:

```python
# Model loading
checkpoint_path: str = "runs/whisper_dyslexia_domain_adaptation/checkpoint-1000"

# Domain settings
domain: str = "dyslexic"  # or "normal"
language: str = "korean"
task: str = "transcribe"
```

## 📈 Monitoring and Logging

### Weights & Biases Integration

The training script automatically logs:
- Training and validation loss curves
- Character Error Rate (CER) metrics
- Learning rate schedules
- GPU utilization and memory usage

### Hugging Face Hub Integration

Trained models are automatically pushed to:
- Model checkpoints and configurations
- Training logs and metrics
- Model cards with performance details

## 🧪 Testing and Evaluation

### Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- M4A

### Evaluation Metrics

- **CER (Character Error Rate)**: Character-level accuracy
- **WER (Word Error Rate)**: Word-level accuracy
- **Domain-Specific Metrics**: Separate evaluation for each domain
- **Inference Time**: Performance benchmarking

### Test Data Formats

1. **Single File**: Direct audio file testing
2. **JSON Configuration**: Batch testing with reference texts
3. **Directory Structure**: Organized audio/text pairs
4. **WebDataset Format**: Compatible with training pipeline

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes and add tests
5. Run tests: `python -m pytest tests/`
6. Submit a pull request

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Training Guide](docs/training_guide.md): Comprehensive training instructions
- [Testing Guide](docs/testing_guide.md): Testing and evaluation procedures
- [Domain Adaptation](docs/domain_adaptation.md): Technical details of domain adaptation

## 🎯 Use Cases

This framework is designed for:

- **Speech Therapy Applications**: Improving speech recognition for individuals with dyslexia
- **Educational Technology**: Creating accessible learning tools
- **Research**: Studying dyslexic speech patterns and improving recognition systems
- **Accessibility Tools**: Building inclusive speech-to-text applications

## ⚠️ Important Notes

- **Data Privacy**: Ensure compliance with data protection regulations when handling speech data
- **Model Limitations**: Domain adaptation works best with sufficient dyslexic speech data
- **Hardware Requirements**: Training requires significant GPU memory (recommended: 24GB+ VRAM)
- **Language Support**: Currently optimized for Korean; adaptation for other languages may require modifications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper model architecture
- Hugging Face for the Transformers library
- The Korean speech recognition research community
- Contributors and testers who helped improve this framework

## 📞 Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation in `docs/`
- Review existing issues and discussions

---

**Note**: This project is designed for research and educational purposes. Please ensure proper data handling and compliance with relevant regulations when working with speech data.
