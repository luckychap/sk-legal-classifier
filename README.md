# sk-legal-classifier

Multi-label topic classification of **Slovak legal text** using **SlovakBERT**,
trained on **MultiEURLEX** (EuroVoc topics).

This is a working end-to-end pipeline: download → tokenise → fine-tune →
evaluate → predict. It has been run end to end (`train.py --smoke` passes).

Built to run on **old NVIDIA hardware**. It deliberately avoids Triton,
bitsandbytes, Unsloth and FlashAttention, all of which require compute
capability ≥ 6.0/7.0. A Quadro M6000 (Maxwell, CC 5.2) cannot run any of them,
but it runs a small encoder fine-tune perfectly well.

---

## 1. Why this task first

| | |
|---|---|
| Task | Multi-label classification into 21 top-level EuroVoc domains |
| Data | MultiEURLEX, Slovak subset — 22,971 train / 5,000 val / 5,000 test |
| Labels | Gold, from the EU Publications Office (not hand-made) |
| Metric | micro-F1 / macro-F1 on a held-out test split |
| Model | `gerulata/slovakbert` (RoBERTa-base, ~124M params) |

It is the standard benchmark task for multilingual legal classification
(Chalkidis et al., EMNLP 2021), so you can compare your numbers against
published ones. Once the pipeline works you swap the data source for real
Slovak national legislation — same code, same shape.

---

## 2. Setup — devbox (recommended)

The `devbox.json` provides Python 3.12, uv, and — importantly —
`libstdc++.so.6` via Nix, which is the classic thing that breaks PyTorch on a
Nix host.

```bash
devbox install

# on a machine WITHOUT the M6000 (e.g. a laptop, for a quick check):
devbox run setup-cpu

# on the M6000 box:
devbox run setup-cuda

# one-off data prep: ~2.8 GB download, converts Slovak to Parquet
devbox run fetch-data

devbox run check        # confirm torch can see the GPU
devbox run smoke        # tiny end-to-end run, a couple of minutes
devbox run train        # the real run
```

To classify text (note: `devbox run <script>` does not forward arguments, so
call the interpreter directly for anything with flags):

```bash
devbox run -- python predict.py "Tento zákon upravuje ochranu ovzdušia."
echo "Tento zákon upravuje ochranu ovzdušia." | devbox run predict
```

`devbox run setup-cuda` pins `torch==2.7.1` from the CUDA 12.6 wheel index,
falling back to `2.6.0`.

---

## 3. Setup — plain venv (alternative)

### ⚠️ Install torch FIRST, then the requirements

**This ordering matters and is not obvious.** `accelerate` depends on `torch`,
so running `pip install -r requirements.txt` on its own downloads the latest
PyPI torch — which is now a **CUDA 13** build. CUDA 13 dropped `sm_50`–`sm_72`,
so that torch will refuse to run on a Maxwell card. Install torch explicitly
first; pip then leaves it alone because the constraint is already satisfied.

```bash
python3 -m venv .venv
source .venv/bin/activate

# 1. torch FIRST, from the cu126 index
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 2. only then the rest
pip install -r requirements.txt

# 3. data (one-off, ~2.8 GB)
python fetch_data.py --lang sk
```

If 2.7.1 is unavailable for your Python version, `torch==2.6.0` also works.
Newer cu126 builds (2.8–2.14) may work too — `check_env.py` will tell you.

### Confirm the GPU actually works

```bash
python check_env.py
```

This prints the card, its compute capability, VRAM, and — importantly — the
`Compiled archs` list of the installed torch build. If you see:

```
!! sm_52 is NOT in this torch build's arch list.
```

then that wheel cannot run on your GPU and you need an older cu126 build.

---

## 4. About the data step

`fetch_data.py` downloads the upstream MultiEURLEX tarball once and writes
Parquet files plus a label list:

```
data/sk/train.parquet       22,971 rows    ~84 MiB
data/sk/validation.parquet   5,000 rows    ~18 MiB
data/sk/test.parquet         5,000 rows    ~26 MiB
data/sk/labels.json         21 EuroVoc level-1 ids, in order
```

**Why not just `load_dataset("coastalcph/multi_eurlex")`?** Because HuggingFace
removed loading-script support in `datasets` **4.0**, and that dataset only
ships as a Python script (`multi_eurlex.py`). HuggingFace's own dataset viewer
refuses it too ("runs arbitrary python code"), so there is no Parquet branch to
fall back on. Converting once is the reliable fix — the resulting Parquet loads
on any `datasets` version, forever.

`train.py` prefers `data/<config>/*.parquet` and only falls back to the hub if
they are missing (in which case it prints the `fetch_data.py` command).

---

## 5. Run

```bash
# tiny end-to-end sanity check, a couple of minutes
python train.py --smoke

# the real run
python train.py --epochs 8
```

A single SlovakBERT fine-tune is small: expect a few hours on the M6000.
Watch `micro_f1` on the validation split — expect to land in the 0.60–0.70
range, which is roughly what published encoders get on this task.

> Smoke-mode numbers are meaningless for quality (256 examples, 1 epoch → micro-F1
> around 0.25). `--smoke` proves the *pipeline*, not the model.

### Useful flags

```
--epochs 8            number of epochs (early stopping, patience 3)
--batch-size 16       fits easily in 24 GiB
--max-length 512      truncate documents at 512 tokens
--limit-train 5000    quick experiment on a subset
--fp16                halves activation memory. NOT faster on Maxwell.
--workers 2           dataloader processes (you have 4 cores)
--threshold 0.5       sigmoid cutoff for multi-label decisions
```

---

## 6. Classify something

```bash
python predict.py "Tento zákon upravuje podmienky ochrany ovzdušia a emisných limitov."
python predict.py --file predpis.txt
```

```
 prob   hit  topic
-----  ----  -------------------------
0.981   yes  životné prostredie
0.412     -  energia
0.201     -  doprava
```

---

## 7. Files

```
devbox.json             reproducible environment (Python 3.12 + libstdc++ via Nix)
fetch_data.py           one-off MultiEURLEX -> local Parquet converter
check_env.py            verify torch + GPU + compute capability
train.py                fine-tune + evaluate + save
predict.py              run a trained model on new text
requirements.txt        everything EXCEPT torch (see the ordering note above)
data/labels_sk.json     21 EuroVoc IDs -> Slovak + English names
data/sk/*.parquet       converted dataset (created by fetch_data.py)
runs/slovakbert-eurovoc/  output: weights, tokenizer, labels.json, metrics
```

---

## 8. The hardware reality check

Your box: Quadro M6000 24GB (Maxwell, CC 5.2), Xeon E3-1265L V2, 16GB RAM, HDD.

**GPU** — fine. 24 GiB is ample for a 124M-parameter encoder; you could use a
batch size of 64 and still have room. The GPU is not the bottleneck.

**What the GPU cannot do** — the modern stack. Verified constraints:
- Triton requires CC ≥ 7.0 → no Unsloth
- bitsandbytes requires CC ≥ 6.0 for 8/4-bit; Maxwell is deprecated → no QLoRA
- FlashAttention requires CC ≥ 8.0 (Ampere)
- torch cu128/cu129 wheels dropped Maxwell; CUDA 13 dropped `sm_50`–`sm_72`

**RAM (16GB)** — your real constraint. Tokenising 23k documents at 512 tokens
is the memory-heavy step. If you hit OOM, lower `--max-length` to 256 (which
barely hurts this task) or use `--limit-train`.

**CPU (4 cores, 2012)** — tokenisation is CPU-bound. `--workers 2` is about
right; more will thrash 16GB of RAM.

**HDD** — slow I/O. `fetch_data.py` streams, so it needs ~2.8 GB of download
bandwidth but only ~130 MiB of disk for the result. The HF model cache lives in
`~/.cache/huggingface`.

**Observed on this machine (8 cores, CPU-only smoke run)**
```
fetch_data.py               1.3 min   (2.8 GB download)
train.py --smoke            1.8 min   (256 examples, 1 epoch, CPU)
```
On the M6000 the GPU steps will be much faster than CPU; the real run is
dominated by the full 23k-document dataset and 8 epochs.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: libstdc++.so.6` | Running outside devbox on a Nix host. Use `devbox run ...`, or export `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. |
| `Dataset scripts are no longer supported, but found multi_eurlex.py` | `datasets >= 4`. Run `python fetch_data.py --lang sk`. |
| torch is a `+cu130` build / "capability 5.2 not supported" | You installed requirements before torch. Reinstall torch from the cu126 index first. |
| `!! sm_52 is NOT in this torch build's arch list.` | Wrong wheel. Use an older cu126 build (`2.7.1`, `2.6.0`). |
| `Float can't be cast to the desired output type Long` | Labels not converted to float multi-hot. Already handled in `train.py`. |

---

## 10. Where to take it next

1. **Baseline first.** Get `--smoke` green, then a real run, then record your
   micro-F1. Everything after this is measured against that number.
2. **Try `xlm-roberta-base` as an alternative base model** — sometimes beats
   the monolingual model on legal text, and is the same code path.
3. **Swap in Slovak national legislation.** The JÚĽŠ SAV *Korpus právnych
   predpisov v slovenčine* (45M tokens, EuroVoc/IATE-tagged) lets you keep the
   same label space but train on real Slovak law instead of EU law translated
   into Slovak.
4. **Only then consider an LLM.** If classification plateaus, a small
   instruction-tuned model (Qwen3-1.7B) is the next step — still fits in 24GB.

---

## 11. Licensing and caveats

- MultiEURLEX: CC-BY-SA-4.0.
- SlovakBERT: MIT.
- EuroVoc descriptors are from the EU Vocabularies.
- Model output is a **topic tag**, not legal advice. Do not present it as such.