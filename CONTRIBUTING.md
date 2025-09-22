# Contributing to Whisper Domain Adaptation

Thank you for your interest in contributing to the Whisper Domain Adaptation project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** in the `docs/` directory
3. **Verify the issue** with the latest version of the code

When creating an issue, please include:

- **Clear description** of the problem or feature request
- **Steps to reproduce** (for bugs)
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU info)
- **Relevant logs** or error messages

### Suggesting Enhancements

For feature requests:

1. **Check existing issues** for similar requests
2. **Provide clear use cases** and motivation
3. **Consider implementation complexity**
4. **Discuss with maintainers** for large changes

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. **Fork and clone** the repository:
```bash
git clone https://github.com/yourusername/whisper-domain-adaptation.git
cd whisper-domain-adaptation
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools
```

4. **Install pre-commit hooks** (optional but recommended):
```bash
pre-commit install
```

### Development Tools

The project uses several development tools:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing
- **Pre-commit**: Git hooks

## 📝 Code Style Guidelines

### Python Code Style

- Follow **PEP 8** conventions
- Use **type hints** for function parameters and return values
- Write **docstrings** for all public functions and classes
- Keep functions **focused and small**
- Use **meaningful variable names**

### Example Code Style

```python
def transcribe_audio(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_waveform: torch.Tensor,
    language: str = "korean",
    task: str = "transcribe",
    domain: str = "normal"
) -> str:
    """
    Transcribe a single audio waveform with domain adaptation.
    
    Args:
        model: The Whisper model for transcription
        processor: The Whisper processor for feature extraction
        audio_waveform: Audio waveform tensor
        language: Language for transcription (default: korean)
        task: Task type - transcribe or translate (default: transcribe)
        domain: Domain for transcription - normal or dyslexic (default: normal)
    
    Returns:
        Transcribed text string
    
    Raises:
        ValueError: If domain is not 'normal' or 'dyslexic'
    """
    if domain not in ["normal", "dyslexic"]:
        raise ValueError(f"Domain must be 'normal' or 'dyslexic', got {domain}")
    
    # Implementation here...
    return transcription
```

### Documentation Style

- Use **Markdown** for documentation files
- Include **code examples** where helpful
- Keep documentation **up-to-date** with code changes
- Use **clear, concise language**

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for **new functionality**
- Include **edge cases** and error conditions
- Use **descriptive test names**
- Keep tests **independent** and **repeatable**

### Example Test

```python
import pytest
import torch
from src.test_whisper_model import transcribe_audio

def test_transcribe_audio_normal_domain():
    """Test transcription with normal domain."""
    # Setup
    model = load_test_model()
    processor = load_test_processor()
    audio_waveform = create_test_audio()
    
    # Execute
    result = transcribe_audio(
        model, processor, audio_waveform, domain="normal"
    )
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0

def test_transcribe_audio_invalid_domain():
    """Test transcription with invalid domain raises error."""
    model = load_test_model()
    processor = load_test_processor()
    audio_waveform = create_test_audio()
    
    with pytest.raises(ValueError, match="Domain must be"):
        transcribe_audio(model, processor, audio_waveform, domain="invalid")
```

## 📦 Pull Request Process

### Before Submitting

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run tests and linting**:
```bash
pytest
black --check .
flake8 .
mypy .
```

6. **Commit your changes**:
```bash
git add .
git commit -m "Add feature: brief description"
```

### Submitting the PR

1. **Push your branch**:
```bash
git push origin feature/your-feature-name
```

2. **Create a Pull Request** with:
   - **Clear title** describing the change
   - **Detailed description** of what was changed and why
   - **Reference to related issues**
   - **Screenshots** for UI changes
   - **Testing instructions** for reviewers

3. **Wait for review** and address feedback

### PR Guidelines

- **Keep PRs focused** - one feature/fix per PR
- **Write clear commit messages**
- **Update documentation** as needed
- **Add tests** for new functionality
- **Ensure CI passes** before requesting review

## 🏗️ Project Structure

Understanding the project structure helps with contributions:

```
whisper-domain-adaptation/
├── whisper_dyslexia_cross_domain.py    # Main training script
├── test_whisper_model.py               # General testing
├── test_dyslexia_dataset.py            # Dataset-specific testing
├── examples/                           # Example scripts
├── docs/                              # Documentation
├── tests/                             # Test files
├── requirements.txt                    # Dependencies
└── README.md                          # Project overview
```

## 🎯 Areas for Contribution

### High Priority

- **Bug fixes** and performance improvements
- **Additional language support** beyond Korean
- **Better error handling** and user feedback
- **Documentation improvements**

### Medium Priority

- **New domain adaptation techniques**
- **Additional evaluation metrics**
- **Model optimization** for different hardware
- **Integration examples** with other tools

### Low Priority

- **UI/Web interface** for easier usage
- **Additional audio format support**
- **Advanced visualization** tools

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
```bash
python --version
pip list | grep torch
nvidia-smi  # If using GPU
```

2. **Minimal reproduction case**
3. **Expected vs actual behavior**
4. **Error messages** and stack traces
5. **Steps to reproduce**

## 💡 Feature Requests

For feature requests:

1. **Check existing issues** first
2. **Describe the use case** clearly
3. **Explain the expected behavior**
4. **Consider implementation complexity**
5. **Discuss with maintainers** for large features

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check `docs/` directory first
- **Code Comments**: Look at existing code for examples

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** acknowledgments
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📋 Checklist for Contributors

Before submitting:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Clear commit messages
- [ ] PR description is complete
- [ ] Related issues referenced

## 🤔 Questions?

If you have questions about contributing:

1. **Check the documentation** in `docs/`
2. **Search existing issues** and discussions
3. **Open a new issue** with the "question" label
4. **Join discussions** in GitHub Discussions

Thank you for contributing to the Whisper Domain Adaptation project! 🎉
