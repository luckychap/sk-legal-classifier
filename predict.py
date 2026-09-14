#!/usr/bin/env python3
"""
Classify Slovak legal text with a model trained by train.py.

    python predict.py "Tento zákon upravuje podmienky ochrany životného prostredia..."
    python predict.py --file nejaky_predpis.txt
    echo "text" | python predict.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="text to classify (or use --file/stdin)")
    p.add_argument("--file", help="read text from a file instead")
    p.add_argument("--model", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "runs", "slovakbert-eurovoc"),
        help="path to a trained model directory")
    p.add_argument("--top", type=int, default=5, help="show top-N labels")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the decision threshold stored with the model")
    return p.parse_args()


def read_text(a: argparse.Namespace) -> str:
    if a.file:
        with open(a.file, encoding="utf-8") as fh:
            return fh.read()
    if a.text:
        return a.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("no input: pass text, --file, or pipe via stdin")


def main() -> int:
    a = parse_args()
    text = read_text(a).strip()
    if not text:
        raise SystemExit("input text is empty")

    meta_path = os.path.join(a.model, "labels.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"no labels.json in {a.model} - train a model first")
    meta = json.load(open(meta_path, encoding="utf-8"))
    display = meta["label_display_sk"]
    threshold = a.threshold if a.threshold is not None else meta.get("threshold", 0.5)
    max_length = meta.get("max_length", 512)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device).eval()

    batch = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(**batch).logits[0].float().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    order = np.argsort(-probs)
    print(f"\ntext: {text[:90]}{'...' if len(text) > 90 else ''}\n")
    print(f"{'prob':>6}  {'hit':>4}  topic")
    print("-" * 52)
    for i in order[: a.top]:
        hit = "yes" if probs[i] >= threshold else "-"
        print(f"{probs[i]:6.3f}  {hit:>4}  {display[str(i)]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
