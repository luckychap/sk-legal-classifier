#!/usr/bin/env python3
"""
Download MultiEURLEX and convert the Slovak subset to plain Parquet.

WHY THIS EXISTS
  `datasets` 4.0 removed support for loading scripts, and coastalcph/multi_eurlex
  only ships as a Python script (multi_eurlex.py). So `load_dataset(...)` no
  longer works on modern `datasets`, and HuggingFace's dataset viewer can't
  convert it either (it refuses datasets that "run arbitrary python code").

  This downloads the upstream tarball ONCE, extracts just the language you want,
  and writes Parquet. Parquet loads on every `datasets` version, forever.

    python fetch_data.py                 # Slovak (default) -> data/sk/
    python fetch_data.py --lang sk

  Output:
    data/sk/train.parquet
    data/sk/validation.parquet
    data/sk/test.parquet
    data/sk/labels.json      ordered EuroVoc level-1 concept ids

  Expect a ~2.8 GB download (the tarball contains all 23 languages).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request

URL = ("https://huggingface.co/datasets/coastalcph/multi_eurlex/"
       "resolve/main/data/multi_eurlex.tar.gz")

# archive member -> output split name
MEMBERS = {"train.jsonl": "train", "dev.jsonl": "validation", "test.jsonl": "test"}

LABEL_LEVEL = "level_1"

# EuroVoc level-1 concept ids, in the order the dataset defines them.
# (Extracted from multi_eurlex.py: _CONCEPTS["level_1"])
LEVEL_1 = [
    "100149", "100160", "100148", "100147", "100152", "100143", "100156",
    "100158", "100154", "100153", "100142", "100145", "100150", "100162",
    "100159", "100144", "100151", "100157", "100161", "100146", "100155",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ProgressReader(io.RawIOBase):
    """Wrap an HTTP response so we can report download progress."""

    def __init__(self, resp):
        self._resp = resp
        self._total = int(resp.headers.get("Content-Length") or 0)
        self._read = 0
        self._last = 0.0

    def readable(self) -> bool:
        return True

    def readinto(self, b):
        chunk = self._resp.read(len(b))
        if not chunk:
            return 0
        n = len(chunk)
        b[:n] = chunk
        self._read += n
        now = time.time()
        if now - self._last > 2:
            self._last = now
            if self._total:
                pct = 100 * self._read / self._total
                log(f"  downloaded {self._read / 2**20:,.0f} / "
                    f"{self._total / 2**20:,.0f} MiB ({pct:.1f}%)")
            else:
                log(f"  downloaded {self._read / 2**20:,.0f} MiB")
        return n


def fetch(lang: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    idx = {c: i for i, c in enumerate(LEVEL_1)}

    log(f"streaming {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "sk-legal-classifier"})
    resp = urllib.request.urlopen(req)

    written = {}
    t0 = time.time()

    with tarfile.open(fileobj=ProgressReader(resp), mode="r|gz") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            split = MEMBERS.get(name)
            if split is None or not member.isfile():
                continue

            log(f"extracting {member.name} -> {split} [{lang}]")
            fh = tar.extractfile(member)
            rows, skipped = [], 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = (data.get("text") or {}).get(lang)
                if not text:
                    skipped += 1
                    continue
                concepts = (data.get("eurovoc_concepts") or {}).get(LABEL_LEVEL) or []
                labels = sorted({idx[c] for c in concepts if c in idx})
                if not labels:
                    skipped += 1
                    continue
                rows.append({
                    "celex_id": data.get("celex_id") or "",
                    "text": text,
                    "labels": labels,
                })

            log(f"  {len(rows):,} rows kept, {skipped:,} skipped (no {lang} text)")

            from datasets import Dataset, Features, Sequence, Value

            features = Features({
                "celex_id": Value("string"),
                "text": Value("string"),
                "labels": Sequence(Value("int8")),
            })
            path = os.path.join(out_dir, f"{split}.parquet")
            Dataset.from_list(rows, features=features).to_parquet(path)
            written[split] = (len(rows), os.path.getsize(path))
            del rows

    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump({"label_level": LABEL_LEVEL, "label_ids": LEVEL_1}, fh, indent=2)

    log(f"done in {(time.time() - t0) / 60:.1f} min")
    for split, (n, size) in written.items():
        log(f"  {split:10s} {n:7,} rows  {size / 2**20:8.1f} MiB")
    log(f"wrote {out_dir}/labels.json")
    if "train" not in written:
        log("ERROR: train split not found in the archive")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", default="sk", help="language ISO code (default: sk)")
    p.add_argument("--out", default=None, help="output dir (default: data/<lang>)")
    a = p.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", a.lang)
    return fetch(a.lang, out)


if __name__ == "__main__":
    raise SystemExit(main())
