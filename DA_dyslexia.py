"""
Whisper Domain Adaptation for Dyslexic vs. Typical Speech

What
- Goal: adapt a general-purpose Whisper checkpoint (large-v3) to better transcribe speech from two domains: dyslexic and typical speakers. The model is prompted with a domain token per example and fine-tuned with selective unfreezing for efficient adaptation.

Why
- Dyslexic speech can differ in prosody, fluency, and error patterns; out-of-the-box ASR models may underperform. Domain adaptation aims to reduce character error rate (CER) while preserving performance on typical speech, with minimal compute by updating only a subset of layers.

How (high level)
- Data: WebDataset shards for both domains (train/train_normal and val splits) with metadata-based sample counts.
- Text/Audio preprocessing: robust audio decoding via soundfile, resampling to 16 kHz, mono mix, light verbatim text normalization (NFC + whitespace collapse only).
- Domain conditioning: prepend special tokens "<|domain:dyslexic|>" or "<|domain:normal|>" to labels at collate time; extend tokenizer and resize embeddings.
- Selective fine-tuning: freeze most model weights; unfreeze last 4 encoder blocks and attention/MLP submodules of the last 4 decoder blocks. Enable dropout/activation_dropout for regularization.
- Training: Hugging Face Seq2SeqTrainer with cosine LR, warmup, label smoothing, bf16, persistent dataloader workers, periodic eval and checkpointing. Metric: CER via evaluate.load("cer"). Optional logging to W&B and model push to Hugging Face Hub.

Key entry points
- Dataset pipeline: build_datasets() assembles shards, infers domain labels, and emits Whisper input features.
- Collator: DataCollatorSpeechSeq2SeqDomain() pads features, injects domain tokens into labels, and builds decoder inputs.
- Trainer: build_trainer() wires processor/model/collator/metrics, applies selective unfreezing, decoding config, and training args.

Operational notes
- Configure via CLI flags or environment; see parse_args() for the full surface.
- To run: python whisper_domain_adaptation.py --dataset-root <root> --output-dir runs/whisper_dyslexia --run-name whisper_dyslexia_r1 --login-wandb --login-hf
- A visual overview of the pipeline lives at docs/whisper_dyslexia_domain_adaptation.svg
- A manager-ready report with rationale, settings, and next steps is in docs/whisper_domain_adaptation_report.md
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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
    dataset_root: str = "/data2/irfan/dataset/UTAP"
    samples_per_shard: int = 1000  # Used for fallback estimation if metadata unavailable
    target_sr: int = 16000

    # =============================================================================
    # Model Configuration
    # =============================================================================
    base_model: str = "runs/whisper_dyslexia/checkpoint-59000"

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
    num_train_epochs: int = 50
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

def norm_verbatim(text: str | None) -> str:
    """Minimal normalization to preserve fillers/punctuation while ensuring NFC + collapsed whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_domain_label(sample: Dict[str, Any]) -> str:
    """All samples are treated as dyslexic in this variant."""
    return "dyslexic"


def decode_text_field(text_field: Union[str, bytes]) -> str:
    if isinstance(text_field, bytes):
        return text_field.decode("utf-8")
    if isinstance(text_field, str):
        return text_field
    return str(text_field)


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

    train_dir = os.path.join(cfg.dataset_root, "train")
    train_urls = sorted(glob.glob(os.path.join(train_dir, "*.tar"))) if os.path.isdir(train_dir) else []
    if not train_urls:
        raise FileNotFoundError(f"No dyslexic training shards found under {train_dir}")

    val_dirs = [
        os.path.join(cfg.dataset_root, "val"),
        os.path.join(cfg.dataset_root, "val_small"),
        os.path.join(cfg.dataset_root, "validation"),
    ]
    val_urls: List[str] = []
    val_counts: List[int] = []
    for val_dir in val_dirs:
        if not os.path.isdir(val_dir):
            continue
        urls = sorted(glob.glob(os.path.join(val_dir, "*.tar")))
        if not urls:
            continue
        metadata_path = os.path.join(val_dir, "metadata.json")
        samples = cfg.samples_per_shard * len(urls)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                samples = int(metadata.get("total_samples", samples))
            except Exception as exc:
                print(f"Warning: failed to parse {metadata_path}: {exc}")
        print(f"[val:dyslexic] shards={len(urls)} samples≈{samples} (metadata)")
        val_urls.extend(urls)
        val_counts.append(samples)

    if not val_urls:
        raise FileNotFoundError(f"No dyslexic validation shards found under {cfg.dataset_root}")

    metadata_path = os.path.join(train_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            train_samples = int(metadata.get("total_samples", cfg.samples_per_shard * len(train_urls)))
        except Exception as exc:
            print(f"Warning: failed to parse {metadata_path}: {exc}")
            train_samples = cfg.samples_per_shard * len(train_urls)
    else:
        train_samples = cfg.samples_per_shard * len(train_urls)

    num_val_samples = sum(val_counts) if val_counts else cfg.samples_per_shard * len(val_urls)

    print(f"[train:dyslexic] shards={len(train_urls)} samples≈{train_samples}")
    print(f"Total train samples≈{train_samples} (dyslexic only)")
    print(f"Total val samples≈{num_val_samples} (dyslexic only)")

    num_train_samples = train_samples
    num_val_samples = num_val_samples

    def prepare_dataset(sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_waveform = preprocess_audio(sample["wav"], cfg.target_sr)
        normalized_text = norm_verbatim(decode_text_field(sample["txt"]))
        batch: Dict[str, Any] = {}
        batch["input_features"] = processor.feature_extractor(
            audio_waveform, sampling_rate=cfg.target_sr
        ).input_features[0]
        batch["text"] = normalized_text
        batch["domain"] = "dyslexic"
        return batch

    train_dataset = (
        wds.WebDataset(train_urls, shardshuffle=True)
        .shuffle(1000)
        .map(prepare_dataset)
        .with_length(num_train_samples)
    )
    val_dataset = (
        wds.WebDataset(val_urls, shardshuffle=False)
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
class DataCollatorSpeechSeq2SeqDomain:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", padding="longest"
        )

        if "attention_mask" not in batch:
            lengths = torch.tensor(
                [f["input_features"].shape[-1] for f in features],
                dtype=torch.long,
            )
            max_len = int(lengths.max().item()) if lengths.numel() else 0
            attention_mask = torch.zeros(len(features), max_len, dtype=torch.long)
            for idx, length in enumerate(lengths.tolist()):
                attention_mask[idx, :length] = 1
            batch["attention_mask"] = attention_mask

        label_texts = []
        for f in features:
            domain = f.get("domain", "normal")
            token = "<|domain:dyslexic|>" if domain=="dyslexic" else "<|domain:normal|>"
            label_texts.append(token + " " + f["text"])

        labels_batch = self.processor.tokenizer(
            text_target=label_texts, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:,0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:,1:]

        batch["labels"] = labels
        decoder_input_ids = labels.new_full(labels.shape, self.processor.tokenizer.pad_token_id)
        decoder_input_ids[:, 1:] = torch.where(
            labels[:, :-1] == -100,
            self.processor.tokenizer.pad_token_id,
            labels[:, :-1],
        )
        decoder_input_ids[:, 0] = self.processor.tokenizer.bos_token_id
        batch["decoder_input_ids"] = decoder_input_ids
        return batch



class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer to guard against Whisper seek fallback returning empty sequences."""

    def __init__(self, *args, forced_decoder_ids=None, generation_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_decoder_ids = forced_decoder_ids
        self._extra_gen_kwargs = generation_kwargs or {}

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)
        has_labels = "labels" in inputs
        labels = inputs.get("labels") if has_labels else None

        loss = None
        if has_labels:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss.detach()

        generation_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "decoder_input_ids"}}
        gen_kwargs = dict(self._extra_gen_kwargs)
        try:
            with torch.no_grad():
                generated_tokens = model.generate(
                    **generation_inputs,
                    forced_decoder_ids=self._forced_decoder_ids,
                    **gen_kwargs,
                )
        except IndexError:
            gen_kwargs.update(
                {
                    "num_beams": 1,
                    "do_sample": False,
                    "suppress_tokens": None,
                    "begin_suppress_tokens": None,
                    "forced_decoder_ids": self._forced_decoder_ids,
                }
            )
            with torch.no_grad():
                generated_tokens = model.generate(**generation_inputs, **gen_kwargs)

        if generated_tokens is None or generated_tokens.numel() == 0:
            batch_size = next(iter(generation_inputs.values())).size(0)
            eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            generated_tokens = torch.full(
                (batch_size, 1),
                eos_id,
                dtype=torch.long,
                device=model.device,
            )

        return (loss, generated_tokens, labels)



def build_metrics(processor: WhisperProcessor):
    import evaluate
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [norm_verbatim(s) for s in pred_str]
        label_str = [norm_verbatim(s) for s in label_str]

        # If eval dataset includes 'domain', compute per-domain CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}
    return compute_metrics


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------

def build_trainer(cfg: Config) -> Seq2SeqTrainer:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Using base checkpoint: {cfg.base_model}")

    processor = WhisperProcessor.from_pretrained(
        cfg.base_model, language="korean", task="transcribe"
    )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token



    train_dataset, val_dataset, n_train, n_val = build_datasets(cfg, processor)

    data_collator = DataCollatorSpeechSeq2SeqDomain(processor=processor)
    compute_metrics = build_metrics(processor)

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.base_model, use_safetensors=True
    )

    # --- Domain adaptation: freeze most layers, unfreeze last few ---
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last 4 encoder blocks
    for block in model.model.encoder.layers[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    # unfreeze last 4 decoder blocks (attention + MLP only)
    for block in model.model.decoder.layers[-5:]:
        for name, p in block.named_parameters():
            if any(k in name for k in ["q_proj","k_proj","v_proj","out_proj","fc1","fc2","final_layer_norm"]):
                p.requires_grad = True
    

        # Add domain tokens
    special_tokens = ["<|domain:normal|>", "<|domain:dyslexic|>"]
    processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    # Allow gradients only on the newly added domain token embeddings.
    embedding_weight = model.get_input_embeddings().weight
    embedding_weight.requires_grad_(True)
    domain_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
    freeze_mask = torch.ones_like(embedding_weight, dtype=torch.bool)
    freeze_mask[domain_token_ids] = False

    def restrict_embedding_grad(grad: torch.Tensor) -> torch.Tensor:
        mask = freeze_mask
        if mask.device != grad.device:
            mask = mask.to(grad.device)
        return grad.masked_fill(mask, 0)

    embedding_weight.register_hook(restrict_embedding_grad)

    model.generation_config.task = "transcribe"
    model.generation_config.language = "korean"

    # Clear the deprecated parameters to avoid conflicts
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # Configure other generation parameters
    model.config.suppress_tokens = None
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.num_beams = 1
    model.generation_config.length_penalty = 1.0
    model.config.use_cache = False

    model.config.dropout = 0.2              
    model.config.activation_dropout = 0.2    

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
        push_to_hub=False,
        hub_model_id=hub_model_id,
        hub_private_repo=cfg.hub_private_repo,
        label_smoothing_factor=0.1,
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")
    generation_kwargs = {
        "num_beams": 1,
        "do_sample": False,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
    }

    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
        processing_class=processor,
        forced_decoder_ids=forced_decoder_ids,
        generation_kwargs=generation_kwargs,
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

    p.add_argument("--dataset-root", type=str, default="/data2/irfan/dataset/UTAP",
                   help="Root directory containing train/ and val_small/ directories")
    p.add_argument("--samples-per-shard", type=int, default=1000,
                   help="Fallback estimate for samples per shard (used if metadata unavailable)")
    p.add_argument("--target-sr", type=int, default=16000,
                   help="Target sample rate for audio preprocessing")
    p.add_argument("--dyslexic-oversample-factor", type=int, default=1,
                   help="Integer oversample factor for dyslexic shards (1 = no oversampling)")
    p.add_argument("--balance-domains", action="store_true",
                   help="Automatically oversample dyslexic shards to roughly match normal sample count")
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    p.add_argument("--base-model", type=str, 
                   default="runs/whisper_dyslexia/checkpoint-59000",
                   help="Base Whisper model to fine-tune")
    
    # =============================================================================
    # Training Configuration
    # =============================================================================
    # Run identification
    p.add_argument("--output-dir", type=str, default="runs/whisper_dyslexia",
                   help="Directory to save training outputs")
    p.add_argument("--run-name", type=str, default="whisper_dyslexia_r1",
                   help="Name for this training run")
    
    # Training hyperparameters
    p.add_argument("--per-device-train-batch-size", type=int, default=16,
                   help="Training batch size per device")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Number of gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=5e-5,
                   help="Learning rate for training")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Number of warmup steps")
    p.add_argument("--num-train-epochs", type=int, default=50,
                   help="Number of training epochs")
    p.add_argument("--lr-scheduler-type", type=str, default="cosine",
                   help="Learning rate scheduler type")
    
    # Evaluation and checkpointing
    p.add_argument("--eval-steps", type=int, default=500,
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
    p.add_argument("--num-workers", type=int, default=10,
                   help="Number of dataloader workers")
    p.add_argument("--persistent-workers", action="store_true",
                   help="Use persistent dataloader workers")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    p.set_defaults(persistent_workers=True)
    p.add_argument("--remove-unused-columns", action="store_true",
                   help="Remove unused columns from dataset")
    
    # Logging
    p.add_argument("--logging-steps", type=int, default=500,
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

    if cfg.push_to_hub:
        hub_model_id = trainer.args.hub_model_id or cfg.hub_model_id
        print("\n" + "=" * 60)
        print("🚀 Training completed! Preparing to push best model to Hugging Face Hub...")
        print(f"📊 Best model metric (cer): {trainer.state.best_metric}")
        if hub_model_id:
            print(f"📁 Hub model ID: {hub_model_id}")
        print("=" * 60 + "\n")
        try:
            trainer.push_to_hub(
                commit_message=f"Fine-tuned Whisper model: {cfg.run_name} (Best CER: {trainer.state.best_metric})",
                blocking=True,
            )
            print("✅ Successfully pushed best model to Hugging Face Hub!")
        except Exception as exc:
            print(f"❌ Failed to push to Hub: {exc}")
            if hub_model_id:
                print("💡 You can manually push later via trainer.push_to_hub().")
    else:
        print("\n" + "=" * 60)
        print("✅ Training completed! Push to hub disabled.")
        print(f"📊 Best model metric (cer): {trainer.state.best_metric}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
