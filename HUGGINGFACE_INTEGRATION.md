# 🤗 Hugging Face Hub Integration for MiddleSenior Training

## ✅ **Yes! Automatic Model Pushing is Enabled**

The training script is configured to automatically push the **best checkpoint** to Hugging Face Hub during training.

## 📤 **Model Repository Details**

- **Hub Model ID**: `braindeck/whisper-middlesenior-normal-v1`
- **Repository Type**: Private (default)
- **Auto-Push**: Enabled for best checkpoint
- **Push Timing**: At the end of training (best model based on CER)

## 🔧 **Configuration Summary**

### Built-in Hub Integration
```python
# Default settings in the training script:
push_to_hub=True                                    # Auto-push enabled
hub_model_id="braindeck/whisper-middlesenior-normal-v1"  # Repository name
hub_private_repo=True                               # Private repository
load_best_model_at_end=True                        # Load best checkpoint
metric_for_best_model="cer"                        # Best model selection
```

### Training Command Flags
```bash
--push-to-hub                                       # Enable pushing
--login-hf                                         # Auto-login to HF Hub
--hub-model-id braindeck/whisper-middlesenior-normal-v1  # Repository name
```

## 🔑 **Authentication Setup**

### Option 1: Hugging Face CLI (Recommended)
```bash
# Install if not already installed
pip install huggingface_hub

# Login interactively
huggingface-cli login
```

### Option 2: Environment Variable
```bash
# Set your token as environment variable
export HF_TOKEN="your_huggingface_token_here"
```

### Option 3: Token File
```bash
# The script will automatically detect ~/.huggingface/token
echo "your_token" > ~/.huggingface/token
```

## 📋 **What Gets Pushed**

### Model Files
- ✅ **Model weights**: Best checkpoint based on lowest CER
- ✅ **Tokenizer**: Updated with domain tokens (`<|domain:normal|>`)
- ✅ **Configuration**: Model config with domain adaptation settings
- ✅ **Training args**: Complete training configuration

### Repository Structure
```
braindeck/whisper-middlesenior-normal-v1/
├── config.json                    # Model configuration
├── model.safetensors              # Model weights (safetensors format)
├── tokenizer.json                 # Tokenizer with domain tokens
├── tokenizer_config.json          # Tokenizer configuration
├── training_args.bin              # Training arguments
├── README.md                      # Auto-generated model card
└── ...                           # Other training artifacts
```

## 🎯 **Benefits of Hub Integration**

1. **Automatic Backup**: Best model automatically saved to cloud
2. **Version Control**: Track different training runs and experiments
3. **Easy Sharing**: Share model with team or make public later
4. **Direct Loading**: Load model directly from Hub for inference
5. **Model Card**: Automatic documentation generation

## 🔄 **Training Workflow with Hub Push**

1. **Training Starts**: Model begins training on middlesenior dataset
2. **Periodic Evaluation**: Model evaluated every 1000 steps
3. **Best Checkpoint Tracking**: System tracks checkpoint with lowest CER
4. **Training Completion**: Training finishes after 100 epochs
5. **Best Model Loading**: System loads the best checkpoint
6. **Hub Push**: Best model automatically pushed to `braindeck/whisper-middlesenior-normal-v1`
7. **Repository Creation**: Private repository created with model files

## 📚 **Using the Pushed Model**

After training, you can load the model directly from Hugging Face Hub:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load the trained model
model = WhisperForConditionalGeneration.from_pretrained(
    "braindeck/whisper-middlesenior-normal-v1"
)
processor = WhisperProcessor.from_pretrained(
    "braindeck/whisper-middlesenior-normal-v1"
)
```

## ⚠️ **Important Notes**

- **Private Repository**: Default is private - change `--hub-private-repo` to make public
- **Authentication Required**: Must be logged in to Hugging Face Hub
- **Storage Limits**: Check your Hub storage limits for large models
- **Model Size**: ~6GB for whisper-large-v3 based model

**The training script handles everything automatically - just ensure you're authenticated with Hugging Face!** 🚀