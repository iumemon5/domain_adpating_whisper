# 🚀 Quick Start Guide: MiddleSenior Training with Conda Environment

## ✅ Environment Verified!
Your setup is complete and ready for training:
- ✅ Conda environment: `whisperx` activated
- ✅ All required packages installed
- ✅ MiddleSenior dataset found and accessible
- ✅ Training scripts configured

## 🎯 Ready to Train - Choose Your Method:

### Method 1: One-Click Training (Recommended)
```bash
cd /home/braindeck/ssd/irfan/projects/whisper_domain_adaptation
./train_middlesenior.sh
```

### Method 2: Python Script with Environment Check
```bash
cd /home/braindeck/ssd/irfan/projects/whisper_domain_adaptation
conda activate whisperx
python train_middlesenior.py
```

## 📊 Training Configuration Summary
- **Dataset**: `/home/braindeck/ssd/irfan/dataset/middlesenior_dataset`
- **Environment**: `whisperx` conda environment
- **Output**: `runs/whisper_middlesenior_normal`
- **Epochs**: 100 (reduced for pre-training)
- **Domain**: Normal speech with `<|domain:normal|>` tokens
- **Strategy**: Selective layer unfreezing (last 4 layers only)

## 🔍 What Will Happen
1. **Automatic Environment Activation**: Script activates `whisperx`
2. **Domain-Aware Training**: Uses `<|domain:normal|>` tokens for all data
3. **Efficient Training**: Only trains last 4 encoder/decoder layers
4. **Progress Monitoring**: WandB logging enabled for real-time tracking
5. **Checkpoint Saving**: Model saved every 1000 steps

## 📈 Expected Timeline
- **Training Duration**: ~2-4 hours (depending on GPU)
- **Checkpoints**: Every 1000 steps
- **Evaluation**: Every 1000 steps
- **Final Model**: Best checkpoint based on CER

## 🎯 This Training Sets You Up For:
1. **Strong Foundation**: Domain-aware model trained on normal Korean speech
2. **Next Phase**: Perfect base for dyslexic domain adaptation
3. **Comparison Baseline**: Compare against original whisper-large-v3
4. **Production Ready**: Can be used as-is for normal speech recognition

**Ready to start? Just run: `./train_middlesenior.sh`** 🚀