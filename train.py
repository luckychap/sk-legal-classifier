#!/usr/bin/env python3
"""
Fine-tune SlovakBERT for multi-label EuroVoc topic classification of Slovak
legal text (MultiEURLEX, Slovak subset).

WHY THIS SHAPE
  This is deliberately plain PyTorch + transformers. It does NOT use
  Triton, bitsandbytes, Unsloth or FlashAttention, because those require
  compute capability >= 6.0/7.0 and an NVIDIA Quadro M6000 is Maxwell (5.2).
  A small encoder fine-tunes perfectly well without them.

QUICK START
  python fetch_data.py            # once: 2.8 GB download -> data/sk/*.parquet
  python check_env.py             # confirm the GPU works
  python train.py --smoke         # ~2 min end-to-end sanity check
  python train.py --epochs 8      # the real run

DATA
  MultiEURLEX (Chalkidis et al., EMNLP 2021), Slovak subset:
    train 22,971 | validation 5,000 | test 5,000
    labels: 21 top-level EuroVoc domains, multi-label

  The data is read from local Parquet produced by fetch_data.py. That matters:
  HuggingFace removed loading-script support in `datasets` 4.0 and multi_eurlex
  only ships as a script, so `load_dataset("coastalcph/multi_eurlex")` no longer
  works on modern `datasets`.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time

import numpy as np
import torch

import transformers
from datasets import Features, Sequence, Value, load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_FILE = os.path.join(HERE, "data", "labels_sk.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="gerulata/slovakbert",
                   help="HF model id or local path")
    p.add_argument("--dataset", default="coastalcph/multi_eurlex",
                   help="HF dataset id (only used if no local Parquet exists)")
    p.add_argument("--config", default="sk",
                   help="language config; also the local data dir name (data/<config>)")
    p.add_argument("--output", default=os.path.join(HERE, "runs", "slovakbert-eurovoc"))
    p.add_argument("--epochs", type=float, default=8.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="sigmoid decision threshold for multi-label output")
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true",
                   help="AMP fp16. No speedup on Maxwell, but halves activation "
                        "memory. Leave off unless you hit OOM.")
    p.add_argument("--limit-train", type=int, default=0,
                   help="use only N training rows (0 = all)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset, 1 epoch, short sequences - proves the pipeline")
    return p.parse_args()


def build_training_arguments(kwargs: dict, total_steps: int) -> TrainingArguments:
    """Build TrainingArguments against whatever the installed version accepts.

    transformers renamed/removed a few knobs over time:
      evaluation_strategy -> eval_strategy   (4.46+)
      warmup_ratio        -> removed in v5 (warmup_steps only)
      overwrite_output_dir-> removed in v5
    Rather than pin names, ask the installed library what it takes.
    """
    sig = inspect.signature(TrainingArguments.__init__).parameters

    # v5 dropped warmup_ratio; express the same intent with warmup_steps.
    if "warmup_ratio" not in sig and "warmup_ratio" in kwargs:
        ratio = kwargs.pop("warmup_ratio")
        if total_steps > 0:
            kwargs["warmup_steps"] = max(1, int(ratio * total_steps))

    accepted = {k: v for k, v in kwargs.items() if k in sig}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        log(f"note: TrainingArguments ignored unsupported keys: {dropped}")
    return TrainingArguments(**accepted)


def build_trainer(model, args, tokenizer, collator, train_ds, eval_ds, compute_metrics):
    """Handle the tokenizer -> processing_class rename across versions."""
    sig = inspect.signature(Trainer.__init__).parameters
    common = dict(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    if "processing_class" in sig:
        common["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        common["tokenizer"] = tokenizer
    return Trainer(**common)


def main() -> int:
    a = parse_args()

    if a.smoke:
        a.epochs = 1.0
        a.max_length = min(a.max_length, 128)
        a.batch_size = min(a.batch_size, 8)
        a.limit_train = a.limit_train or 256
        log("SMOKE MODE: tiny subset, 1 epoch, short sequences")

    set_seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} | transformers {transformers.__version__} | device={device}")
    if device == "cpu":
        log("WARNING: running on CPU. Fine for a smoke test, far too slow for a real run.")
    else:
        p = torch.cuda.get_device_properties(0)
        log(f"GPU: {p.name}  CC {p.major}.{p.minor}  {p.total_memory / 2**30:.1f} GiB")

    # ---------------------------------------------------------------- data
    # Prefer locally-converted Parquet (see fetch_data.py). HuggingFace's
    # datasets>=4 cannot load script-based datasets, and multi_eurlex is
    # script-only, so the local Parquet is the reliable path.
    local = os.path.join(HERE, "data", a.config)
    label_ids = None
    local_labels = os.path.join(local, "labels.json")
    if os.path.exists(local_labels):
        label_ids = json.load(open(local_labels, encoding="utf-8"))["label_ids"]

    if os.path.exists(os.path.join(local, "train.parquet")):
        files = {s: os.path.join(local, f"{s}.parquet")
                 for s in ("train", "validation", "test")}
        missing = [s for s, p in files.items() if not os.path.exists(p)]
        if missing:
            log(f"ERROR: missing parquet for {missing} in {local}")
            return 2
        log(f"loading local Parquet from {local}")
        raw = load_dataset("parquet", data_files=files)
    else:
        log(f"no local data in {local}; trying the HF hub")
        try:
            raw = load_dataset(a.dataset, a.config)
        except Exception as exc:
            log(f"hub load failed: {exc}")
            log(f"FIX: run   python fetch_data.py --lang {a.config}")
            return 2

    if label_ids is None:
        label_ids = list(raw["train"].features["labels"].feature.names)
    num_labels = len(label_ids)
    log(f"labels ({num_labels}): {', '.join(label_ids)}")

    descriptors = {}
    if os.path.exists(LABELS_FILE):
        descriptors = json.load(open(LABELS_FILE, encoding="utf-8"))
    label_display = {
        i: descriptors.get(lid, {}).get("sk")
        or descriptors.get(lid, {}).get("en") or lid
        for i, lid in enumerate(label_ids)
    }

    def drop_empty(example):
        return bool(example["text"] and example["text"].strip())

    raw = raw.filter(drop_empty, desc="dropping empty documents")

    # Keep class coverage statistics so we can sanity-check the split.
    # NOTE: label lists are ragged (1..n labels per doc), so np.array(...)
    # would fail - count explicitly, and do it BEFORE the multi-hot conversion.
    train_label_counts = np.zeros(num_labels, dtype=np.int64)
    for labs in raw["train"]["labels"]:
        for lab in {int(x) for x in labs}:
            train_label_counts[lab] += 1
    log("train label frequencies (top 5): "
        + ", ".join(f"{label_display[i]}={int(train_label_counts[i])}"
                    for i in np.argsort(-train_label_counts)[:5]))

    def to_multihot(example):
        vec = [0.0] * num_labels
        for lab in example["labels"]:
            vec[int(lab)] = 1.0
        example["labels"] = vec
        return example

    # The labels column starts life as Sequence(int8); without overriding the
    # schema here, `map` would cast the float multi-hot vector back to int8 and
    # the BCE loss would fail with "Float can't be cast to Long".
    multihot_features = Features({
        "celex_id": Value("string"),
        "text": Value("string"),
        "labels": Sequence(Value("float32")),
    })
    raw = raw.map(to_multihot, features=multihot_features,
                  desc="encoding labels as multi-hot")

    # ------------------------------------------------------------ tokenizer
    log(f"loading tokenizer {a.model}")
    tokenizer = AutoTokenizer.from_pretrained(a.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=a.max_length)

    # Subset BEFORE tokenising - otherwise a smoke test tokenises all 23k docs.
    keep = ("input_ids", "attention_mask", "labels")
    tok = {}
    for name in ("train", "validation", "test"):
        ds = raw[name]
        if name == "train" and a.limit_train:
            ds = ds.select(range(min(a.limit_train, len(ds))))
        if a.smoke:
            cap = {"train": 256, "validation": 100, "test": 100}[name]
            ds = ds.select(range(min(cap, len(ds))))
        ds = ds.map(tokenize, batched=True, desc=f"tokenizing {name}")
        tok[name] = ds.remove_columns([c for c in ds.column_names if c not in keep])

    train_ds, eval_ds, test_ds = tok["train"], tok["validation"], tok["test"]
    log(f"train={len(train_ds)}  validation={len(eval_ds)}  test={len(test_ds)}")

    # ---------------------------------------------------------------- model
    log(f"loading model {a.model} with {num_labels} labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # -------------------------------------------------------------- metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        preds = (probs >= a.threshold).astype(int)
        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
            "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        }

    # -------------------------------------------------------------- training
    os.makedirs(a.output, exist_ok=True)
    use_fp16 = bool(a.fp16 and device == "cuda")

    total_steps = int(
        a.epochs * math.ceil(len(train_ds) / max(1, a.batch_size * a.grad_accum))
    )
    ta = build_training_arguments(dict(
        output_dir=a.output,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=max(8, a.batch_size * 2),
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        evaluation_strategy="epoch",       # older transformers; filtered out if unknown
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        dataloader_num_workers=a.workers,
        dataloader_pin_memory=(device == "cuda"),
        report_to=[],
        seed=a.seed,
    ), total_steps=total_steps)

    trainer = build_trainer(
        model, ta, tokenizer, DataCollatorWithPadding(tokenizer),
        train_ds, eval_ds, compute_metrics,
    )

    log("training ...")
    t0 = time.time()
    trainer.train()
    log(f"training finished in {(time.time() - t0) / 60:.1f} min")

    # -------------------------------------------------------------- test set
    log("evaluating on the held-out test split ...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    for k, v in sorted(test_metrics.items()):
        log(f"  {k} = {v}")

    # ---------------------------------------------------------------- save
    trainer.save_model(a.output)
    tokenizer.save_pretrained(a.output)
    json.dump(
        {
            "label_ids": label_ids,
            "label_display_sk": {str(i): label_display[i] for i in range(num_labels)},
            "base_model": a.model,
            "config": a.config,
            "threshold": a.threshold,
            "max_length": a.max_length,
        },
        open(os.path.join(a.output, "labels.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    json.dump({k: float(v) for k, v in test_metrics.items()},
              open(os.path.join(a.output, "test_metrics.json"), "w"), indent=2)
    log(f"model + label map saved to {a.output}")

    # per-class detail, so you can see which topics work and which don't
    pred = trainer.predict(test_ds)
    # transformers v5 returns a plain tuple here; earlier versions return a
    # PredictionOutput with .predictions / .label_ids.
    logits = np.asarray(pred.predictions if hasattr(pred, "predictions") else pred[0])
    gold = np.asarray(pred.label_ids if hasattr(pred, "label_ids") else pred[1])
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    preds = (probs >= a.threshold).astype(int)
    per_class = f1_score(gold, preds, average=None, zero_division=0)
    log("per-class F1:")
    for i in np.argsort(-per_class):
        log(f"  {per_class[i]:.3f}  {label_display[i]}")

    if a.smoke:
        log("SMOKE TEST PASSED - the pipeline works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
