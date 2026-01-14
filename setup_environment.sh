#!/bin/bash
# Environment setup and verification script for MiddleSenior training

echo "🔧 Whisper MiddleSenior Training Environment Setup"
echo "=================================================="
echo

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📁 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded"
    echo
fi

# Check conda installation
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install miniconda/anaconda first."
    exit 1
else
    echo "✅ Conda found: $(conda --version)"
fi

# Activate whisperx environment
echo
echo "📦 Activating conda environment: whisperx"
source ~/miniconda3/etc/profile.d/conda.sh

if conda activate whisperx 2>/dev/null; then
    echo "✅ Environment 'whisperx' activated successfully"
    echo "📍 Current environment: $CONDA_DEFAULT_ENV"
else
    echo "❌ Failed to activate 'whisperx' environment"
    echo "Available environments:"
    conda env list
    echo
    echo "💡 If 'whisperx' doesn't exist, create it with:"
    echo "   conda create -n whisperx python=3.9"
    echo "   conda activate whisperx"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo
echo "🔍 Checking Python packages..."

# Check key packages
packages=("torch" "transformers" "datasets" "librosa" "evaluate" "wandb" "huggingface_hub")
missing_packages=()

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package: installed"
    else
        echo "❌ $package: missing"
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo
    echo "❌ Missing packages detected. Please install:"
    echo "   pip install ${missing_packages[*]}"
    echo "   Or install all requirements:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo
echo "🎯 Checking dataset..."
if [ -d "/home/braindeck/ssd/irfan/dataset/middlesenior_dataset" ]; then
    echo "✅ MiddleSenior dataset found"
    echo "📊 Dataset structure:"
    ls -la /home/braindeck/ssd/irfan/dataset/middlesenior_dataset/
else
    echo "❌ MiddleSenior dataset not found at expected location"
    echo "Expected: /home/braindeck/ssd/irfan/dataset/middlesenior_dataset"
    exit 1
fi

echo
echo "🔑 Checking authentication setup..."

# Check for HF_TOKEN environment variable first
if [ ! -z "$HF_TOKEN" ]; then
    echo "✅ HF_TOKEN environment variable found"
elif [ -f "$HOME/.huggingface/token" ]; then
    echo "✅ Hugging Face token file found"
else
    echo "⚠️ Hugging Face authentication not found"
    echo "💡 To set up Hugging Face authentication:"
    echo "   Method 1: huggingface-cli login"
    echo "   Method 2: Set HF_TOKEN environment variable"
    echo "   Method 3: Add HF_TOKEN to .vscode/settings.json"
    echo
    echo "📋 Your token should have 'write' permissions for model repositories"
fi

if [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc"; then
    echo "✅ WandB authentication found"
else
    echo "⚠️ WandB authentication not found"
    echo "💡 To set up WandB authentication:"
    echo "   wandb login"
fi

echo
echo "�🚀 Environment setup complete! Ready to train."
echo
echo "📤 Model will be pushed to: braindeck/whisper-middlesenior-normal-v1"
echo
echo "To start training, run:"
echo "   ./train_middlesenior.sh"
echo "   OR"
echo "   python train_middlesenior.py"