"""
Whisper fine-tuning script (refactored from whisper_v3.ipynb)
- Structured into clear functions
- Configurable via argparse / environment
- Robust audio decoding using soundfile (avoids torchcodec crash)
- Optional W&B and Hugging Face Hub logins
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import (
    Seq2SeqTrainingArguments,
    Trainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def print_env_summary() -> None:
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except Exception:
        pass
    print("Python:", sys.version)
    try:
        print("Torch:", torch.__version__, "CUDA:", getattr(torch.version, "cuda", None))
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Device:", torch.cuda.get_device_name(0))
            print("Capability:", torch.cuda.get_device_capability(0))
    except Exception as e:
        print("Torch not available:", e)
    try:
        import torchaudio  # noqa: F401 (only for version print)
        print("Torchaudio:", torchaudio.__version__)
    except Exception as e:
        print("Torchaudio not available:", e)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    # =============================================================================
    # Dataset Configuration
    # =============================================================================
    dataset_root: str = "dyslexia_dataset_webdataset"
    samples_per_shard: int = 1000  # Used for fallback estimation if metadata unavailable
    target_sr: int = 16000

    # =============================================================================
    # Model Configuration
    # =============================================================================
    base_model: str = "braindeck/whisper_senior_r1"

    # =============================================================================
    # Training Configuration
    # =============================================================================
    # Run identification
    output_dir: str = "runs/whisper_dyslexia"
    run_name: str = "whisper_dyslexia_r1"
    
    # Training hyperparameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    warmup_steps: int = 1000
    num_train_epochs: int = 100
    lr_scheduler_type: str = "cosine"
    
    # Evaluation and checkpointing
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 5
    
    # Performance and optimization
    bf16: bool = True
    dataloader_num_workers: int = 24
    dataloader_persistent_workers: bool = True
    remove_unused_columns: bool = False
    
    # Logging
    logging_steps: int = 500

    # =============================================================================
    # Monitoring and Reporting
    # =============================================================================
    report_to_wandb: bool = True

    # =============================================================================
    # Model Hub Configuration
    # =============================================================================
    push_to_hub: bool = True
    hub_model_id: str | None = None  # default: f"braindeck/{run_name}"
    hub_private_repo: bool = True

    # =============================================================================
    # Reproducibility
    # =============================================================================
    seed: int = 42


# --------------------------------------------------------------------------------------
# Text and Audio Processing
# --------------------------------------------------------------------------------------

def ko_norm(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
    # Returns [1, num_samples] float32 waveform and sample_rate
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data).T
    return waveform, sr


def preprocess_audio(audio_bytes: bytes, target_sr: int) -> torch.Tensor:
    waveform, sample_rate = load_audio_from_bytes(audio_bytes)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        import torchaudio  # defer import; avoids global backend init
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0)


# --------------------------------------------------------------------------------------
# Dataset & Dataloaders
# --------------------------------------------------------------------------------------

def build_datasets(cfg: Config, processor: WhisperProcessor) -> Tuple[Any, Any, int, int]:
    import glob
    import json

    print(f"dataset_root: {cfg.dataset_root}")

    train_urls = sorted(
        glob.glob(os.path.join(cfg.dataset_root, "train", "*.tar"))
    )
    val_urls = sorted(glob.glob(os.path.join(cfg.dataset_root, "val_small", "*.tar")))

    # Load actual metadata to get precise sample counts
    train_metadata_path = os.path.join(cfg.dataset_root, "train", "metadata.json")
    val_metadata_path = os.path.join(cfg.dataset_root, "val_small", "metadata.json")
    

    print(train_metadata_path)
    print(val_metadata_path)



    with open(train_metadata_path, 'r') as f:
        train_metadata = json.load(f)
    with open(val_metadata_path, 'r') as f:
        val_metadata = json.load(f)
    
    num_train_samples = train_metadata["total_samples"]
    num_val_samples = val_metadata["total_samples"]

    def prepare_dataset(sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_waveform = preprocess_audio(sample["wav"], cfg.target_sr)
        normalized_text = ko_norm(sample["txt"].decode("utf-8"))
        batch: Dict[str, Any] = {}
        batch["input_features"] = processor.feature_extractor(
            audio_waveform, sampling_rate=cfg.target_sr
        ).input_features[0]
        batch["text"] = normalized_text
        return batch

    train_dataset = (
        wds.WebDataset(train_urls)
        .shuffle(1000)
        .map(prepare_dataset)
        .with_length(num_train_samples)
    )
    val_dataset = (
        wds.WebDataset(val_urls)
        .map(prepare_dataset)
        .with_length(num_val_samples)
    )

    return train_dataset, val_dataset, num_train_samples, num_val_samples


def build_dataloaders(
    train_dataset: Any,
    val_dataset: Any,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader


# --------------------------------------------------------------------------------------
# Collator & Metrics
# --------------------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [f["text"] for f in features]
        labels_batch = self.processor.tokenizer(
            text_target=label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_metrics(processor: WhisperProcessor):
    import evaluate

    metric = evaluate.load("cer")

    def compute_metrics(pred):
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        pred_ids = np.argmax(logits, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [ko_norm(s) for s in pred_str]
        label_str = [ko_norm(s) for s in label_str]
        cer = metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_metrics


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------

def build_trainer(cfg: Config) -> Trainer:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Using base checkpoint: {cfg.base_model}")

    processor = WhisperProcessor.from_pretrained(
        cfg.base_model, language="korean", task="transcribe"
    )

    train_dataset, val_dataset, n_train, n_val = build_datasets(cfg, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = build_metrics(processor)

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.base_model, use_safetensors=True
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    model.config.dropout = 0.2              # or your chosen value
    model.config.activation_dropout = 0.2    # optionally, for extra regularization

    hub_model_id = cfg.hub_model_id or f"braindeck/{cfg.run_name}"

    report_to = ["wandb"] if cfg.report_to_wandb else []

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        bf16=cfg.bf16,
        predict_with_generate=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_persistent_workers=cfg.dataloader_persistent_workers,
        remove_unused_columns=cfg.remove_unused_columns,
        logging_steps=cfg.logging_steps,
        report_to=report_to,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=hub_model_id,
        hub_private_repo=cfg.hub_private_repo,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # Quick sanity: try to draw one batch (optional)
    try:
        train_loader, _ = build_dataloaders(
            train_dataset, val_dataset, batch_size=8, num_workers=1
        )
        _ = next(iter(train_loader))
        print("Fetched one batch from train_dataloader.")
        print(f"Estimated training samples: {n_train}")
        print(f"Estimated validation samples: {n_val}")
    except Exception as e:
        print("DataLoader fetch skipped/failed:", e)

    return trainer


# --------------------------------------------------------------------------------------
# Logins (Optional)
# --------------------------------------------------------------------------------------

def maybe_login_wandb(enable: bool) -> None:
    if not enable:
        return
    try:
        import wandb
        key = os.getenv("WANDB_API_KEY")
        project = os.getenv("WANDB_PROJECT", "whisper-senior-r1")
        if key:
            wandb.login(key=key)
        else:
            # Falls back to existing credentials; avoids interactive prompt in scripts
            wandb.login(relogin=False)
        os.environ["WANDB_PROJECT"] = project
        print("Weights & Biases: logged in, project=", project)
    except Exception as e:
        print("Weights & Biases login skipped:", e)


def maybe_login_hf(enable: bool) -> None:
    if not enable:
        return
    try:
        from huggingface_hub import login
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            login(token=token, add_to_git_credential=True)
            print("Hugging Face Hub: logged in via token env var")
        else:
            print("Hugging Face Hub: no token in env; skipping login")
    except Exception as e:
        print("Hugging Face Hub login skipped:", e)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Fine-tune Whisper on WebDataset shards")
    
    # =============================================================================
    # Dataset Configuration
    # =============================================================================

    p.add_argument("--dataset-root", type=str, default="dyslexia_dataset_webdataset",
                   help="Root directory containing train/ and val_small/ directories")
    p.add_argument("--samples-per-shard", type=int, default=1000,
                   help="Fallback estimate for samples per shard (used if metadata unavailable)")
    p.add_argument("--target-sr", type=int, default=16000,
                   help="Target sample rate for audio preprocessing")
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    p.add_argument("--base-model", type=str, 
                   default="Suchae/whisper-large-v3-ko-middlesenior-dialect-speech",
                   help="Base Whisper model to fine-tune")
    
    # =============================================================================
    # Training Configuration
    # =============================================================================
    # Run identification
    p.add_argument("--output-dir", type=str, default="runs/whisper_senior",
                   help="Directory to save training outputs")
    p.add_argument("--run-name", type=str, default="whisper_senior_r1",
                   help="Name for this training run")
    
    # Training hyperparameters
    p.add_argument("--per-device-train-batch-size", type=int, default=16,
                   help="Training batch size per device")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Number of gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Learning rate for training")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Number of warmup steps")
    p.add_argument("--num-train-epochs", type=int, default=250,
                   help="Number of training epochs")
    p.add_argument("--lr-scheduler-type", type=str, default="cosine",
                   help="Learning rate scheduler type")
    
    # Evaluation and checkpointing
    p.add_argument("--eval-steps", type=int, default=1000,
                   help="Steps between evaluations")
    p.add_argument("--save-steps", type=int, default=1000,
                   help="Steps between model saves")
    p.add_argument("--save-total-limit", type=int, default=5,
                   help="Maximum number of checkpoints to keep")
    
    # Performance and optimization
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 precision")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.set_defaults(bf16=True)
    p.add_argument("--num-workers", type=int, default=20,
                   help="Number of dataloader workers")
    p.add_argument("--persistent-workers", action="store_true",
                   help="Use persistent dataloader workers")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    p.set_defaults(persistent_workers=True)
    p.add_argument("--remove-unused-columns", action="store_true",
                   help="Remove unused columns from dataset")
    
    # Logging
    p.add_argument("--logging-steps", type=int, default=50,
                   help="Steps between logging")
    
    # =============================================================================
    # Monitoring and Reporting
    # =============================================================================
    p.add_argument("--report-to-wandb", action="store_true",
                   help="Report metrics to Weights & Biases")
    p.add_argument("--no-report-to-wandb", dest="report_to_wandb", action="store_false")
    p.set_defaults(report_to_wandb=True)
    
    # =============================================================================
    # Model Hub Configuration
    # =============================================================================
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push model to Hugging Face Hub")
    p.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    p.set_defaults(push_to_hub=True)
    p.add_argument("--hub-model-id", type=str, default=None,
                   help="Model ID for Hugging Face Hub (default: braindeck/{run_name})")
    p.add_argument("--hub-private-repo", action="store_true",
                   help="Create private repository on Hugging Face Hub")
    p.add_argument("--hub-public-repo", dest="hub_private_repo", action="store_false")
    p.set_defaults(hub_private_repo=True)
    
    # =============================================================================
    # Reproducibility
    # =============================================================================
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    
    # =============================================================================
    # Authentication (Login flags)
    # =============================================================================
    p.add_argument("--login-wandb", action="store_true",
                   help="Login to Weights & Biases")
    p.add_argument("--login-hf", action="store_true",
                   help="Login to Hugging Face Hub")

    a = p.parse_args()
    return Config(
        dataset_root=a.dataset_root,
        samples_per_shard=a.samples_per_shard,
        target_sr=a.target_sr,
        base_model=a.base_model,
        output_dir=a.output_dir,
        run_name=a.run_name,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.learning_rate,
        warmup_steps=a.warmup_steps,
        num_train_epochs=a.num_train_epochs,
        lr_scheduler_type=a.lr_scheduler_type,
        eval_steps=a.eval_steps,
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        bf16=a.bf16,
        dataloader_num_workers=a.num_workers,
        dataloader_persistent_workers=a.persistent_workers,
        remove_unused_columns=a.remove_unused_columns,
        logging_steps=a.logging_steps,
        report_to_wandb=a.report_to_wandb,
        push_to_hub=a.push_to_hub,
        hub_model_id=a.hub_model_id,
        hub_private_repo=a.hub_private_repo,
        seed=a.seed,
    )


def main() -> None:
    print_env_summary()

    cfg = parse_args()
    set_global_seed(cfg.seed)

    # Optional logins (env-based, non-interactive)
    args = sys.argv
    maybe_login_wandb("--login-wandb" in args and cfg.report_to_wandb)
    maybe_login_hf("--login-hf" in args and cfg.push_to_hub)

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
