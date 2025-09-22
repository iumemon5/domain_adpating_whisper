#!/usr/bin/env python3
"""
Example: Basic Training Script
This example shows how to train a domain-adapted Whisper model with minimal configuration.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from whisper_dyslexia_cross_domain import main, parse_args

def example_basic_training():
    """Example of basic training with default settings."""
    print("="*60)
    print("EXAMPLE: Basic Domain-Adapted Whisper Training")
    print("="*60)
    
    # Set up basic arguments
    sys.argv = [
        "whisper_dyslexia_cross_domain.py",
        "--dataset-root", "path/to/your/dataset",
        "--run-name", "example_training",
        "--num-train-epochs", "10",  # Short training for example
        "--eval-steps", "500",
        "--save-steps", "500",
        "--logging-steps", "100"
    ]
    
    print("Training command:")
    print(" ".join(sys.argv[1:]))  # Skip script name
    print()
    
    print("This will:")
    print("- Load the dyslexia dataset from the specified path")
    print("- Train for 10 epochs with domain adaptation")
    print("- Save checkpoints every 500 steps")
    print("- Log progress every 100 steps")
    print("- Use default hyperparameters optimized for dyslexic speech")
    print()
    
    # Uncomment the line below to actually run training
    # main()

def example_advanced_training():
    """Example of advanced training with custom parameters."""
    print("="*60)
    print("EXAMPLE: Advanced Training Configuration")
    print("="*60)
    
    sys.argv = [
        "whisper_dyslexia_cross_domain.py",
        "--dataset-root", "path/to/your/dataset",
        "--base-model", "openai/whisper-large-v3",
        "--run-name", "advanced_example",
        "--learning-rate", "1e-4",
        "--num-train-epochs", "50",
        "--per-device-train-batch-size", "8",
        "--grad-accum", "4",
        "--warmup-steps", "1000",
        "--eval-steps", "1000",
        "--save-steps", "1000",
        "--bf16",  # Use bfloat16 precision
        "--login-wandb",  # Enable Weights & Biases logging
        "--login-hf"  # Enable Hugging Face Hub integration
    ]
    
    print("Advanced training command:")
    print(" ".join(sys.argv[1:]))
    print()
    
    print("This configuration:")
    print("- Uses Whisper Large V3 as base model")
    print("- Higher learning rate (1e-4) for faster convergence")
    print("- Smaller batch size (8) with gradient accumulation (4)")
    print("- Extended warmup (1000 steps)")
    print("- Enables mixed precision training (bfloat16)")
    print("- Integrates with W&B and Hugging Face Hub")
    print()
    
    # Uncomment to run
    # main()

def example_environment_setup():
    """Example of setting up environment variables."""
    print("="*60)
    print("EXAMPLE: Environment Setup")
    print("="*60)
    
    print("Set up environment variables for optimal experience:")
    print()
    print("# Weights & Biases (optional)")
    print("export WANDB_API_KEY='your_wandb_api_key'")
    print("export WANDB_PROJECT='whisper-dyslexia-domain-adaptation'")
    print()
    print("# Hugging Face Hub (optional)")
    print("export HF_TOKEN='your_huggingface_token'")
    print()
    print("# CUDA settings (if using GPU)")
    print("export CUDA_VISIBLE_DEVICES='0'")
    print()
    print("# Dataset path")
    print("export DATASET_ROOT='/path/to/dyslexia_dataset_webdataset'")
    print()

if __name__ == "__main__":
    example_basic_training()
    print()
    example_advanced_training()
    print()
    example_environment_setup()
    
    print("="*60)
    print("To run these examples:")
    print("1. Update the dataset paths to your actual data")
    print("2. Uncomment the main() calls")
    print("3. Ensure you have the required dependencies installed")
    print("4. Run: python examples/sample_training.py")
    print("="*60)
