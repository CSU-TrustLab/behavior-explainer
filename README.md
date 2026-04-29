# Behavior Explainer

Code for the paper: **[Paper Title]**

## Repository Structure

```
behavior-explainer/
├── src/                          # Core source code
│   ├── datasets.py               # RIVAL10 and EuroSAT dataset classes and dataloaders
│   ├── finetune.py               # Fine-tuning ResNet18 and VGG19 on RIVAL10 / EuroSAT
│   ├── train_aligner.py          # Penultimate-layer extraction and CLIP linear aligner training
│   ├── build_vocab.py            # Concept vocabulary generation via CLIP
│   ├── compute_means.py          # Compute and cache CLIP-space mean vectors (image and text)
│   ├── explain.py                # Abductive/contrastive explanation engine (AXp/CXp, XpEnum, NaiveEnum)
│   ├── run_experiment.py         # Run all 9 configurations (3 erasers × 3 algorithms) for one behavior
│   ├── classify_concepts.py      # LLM-as-a-judge: classify vocabulary concepts as RELEVANT/IRRELEVANT
│   └── generalizability_xpsatenum.py  # Run XpEnum + XpSatEnum on 50/50 splits (generalizability)
├── analysis/                     # Metric computation and plotting (no model loading required)
│   ├── avg_xp_count.py           # Metric 1 — average number of explanations per image
│   ├── compute_time.py           # Metric 2 — compute time per image
│   ├── individual_coverage.py    # Metric 3 — individual coverage: table and decay curves
│   ├── max_coverage.py           # Metric 4 — maximum coverage at K (greedy set cover)
│   ├── size_vs_coverage.py       # Metric 5 — explanation size vs individual coverage (violin plot)
│   ├── generalizability.py       # Metric 6 — generalizability at K (IoU of top-K across splits)
│   ├── validity_ratio.py         # Metric 7 — validity ratio (plausibility via LLM relevance)
│   └── rcic.py                   # Metric 8 — relative cumulative individual coverage at length K
├── notebooks/
│   └── behavior_explainer.ipynb  # Reviewer notebook — reproduces all 8 metrics from precomputed CSVs
├── utils/                        # Shared utilities
│   ├── pickler.py                # Compressed binary serialisation (Pickler / CPU_Unpickler)
│   ├── concept_eraser.py         # Concept erasure: ClipOrthoEraser, ClipSpliceEraser, LeaceEraserWrapper
│   └── mhs.py                    # Minimal hitting set solvers (exact, random, hybrid)
├── datasets/                     # Image datasets (not tracked — see per-folder READMEs for download links)
│   ├── RIVAL10/
│   └── EuroSAT/
├── vocabs/                       # Concept vocabularies and reference word lists
├── intermediate_results/         # Pickled tensors, trained models, and precomputed CSVs (not tracked)
├── results/                      # Generated plots and metric summaries
└── tests/                        # Unit tests
```

## Requirements

```bash
pip install -r requirements.txt
# CLIP (OpenAI) must be installed separately:
pip install git+https://github.com/openai/CLIP.git
```

## Reproducing Results

### 1. Fine-tune vision models

```bash
python src/finetune.py
```

Trains ResNet18 and VGG19 on RIVAL10 and EuroSAT and saves models to `intermediate_results/`.
The ResNet × EuroSAT variant requires an SSL4EO pretrained checkpoint (see `src/finetune.py`).

### 2. Train CLIP linear aligners

```bash
python src/train_aligner.py
```

Extracts penultimate-layer representations from each fine-tuned model, trains bidirectional
linear maps to/from CLIP space, and reports round-trip reconstruction quality.

### 3. Compute image and text means

```bash
python src/compute_means.py --model resnet_rival10
```

Computes the mean CLIP-space embeddings for images (aligned and raw) and for
MS-COCO vocabulary words, and caches them in `intermediate_results/`.
Run once per model; subsequent calls load from cache.

### 4. Build concept vocabularies

```bash
python src/build_vocab.py
```

Generates concept word lists for each dataset and saves them to `vocabs/`.
Uses the cached text mean from the previous step to center the concept vectors.

### 5. Run explanation experiments

```bash
python src/run_experiment.py --model resnet_rival10 --behavior 2 --class-idx 4
```

Runs all 9 configurations (Ortho / Splice / LEACE erasure × XpEnum / XpSatEnum / NaiveEnum)
for one behavior, each with a 1-hour timeout.  Results are written to `results/`.
Pass `--help` for the full list of options.

### 6. Classify concepts for the Validity Ratio metric

```bash
export ANTHROPIC_API_KEY=sk-...
python src/classify_concepts.py --dataset rival10 --behavior B26
```

Uses Claude (Anthropic API) to label each concept in the vocabulary as `RELEVANT` or
`IRRELEVANT` for the given behavior.  The result is saved to
`intermediate_results/relevance_rival10_B26.json` and read by Metric 7.
Run once per `(dataset, behavior)` pair; re-run with `--force` to overwrite.

---

## Reproducing Metrics (Reviewer Notebook)

Once the experiments in steps 1–5 have been run and the precomputed CSV files are
present in `intermediate_results/`, **no model loading or GPU is required** to
reproduce the paper's metrics and figures.

### Option A — Jupyter notebook (recommended)

Open `notebooks/behavior_explainer.ipynb` and run all cells.
Set the model, dataset, vocabulary size, and behavior in the *Configuration* cell;
all eight metrics will be computed and saved to `results/` automatically.

```bash
jupyter notebook notebooks/behavior_explainer.ipynb
```

> **Note for Metric 7 (Validity Ratio):** run `src/classify_concepts.py` first
> (step 6 above) to generate the relevance JSON for your behavior.

### Option B — Individual CLI scripts

Each metric can also be run as a standalone script:

| Metric | Script | Description |
|--------|--------|-------------|
| 1 | `analysis/avg_xp_count.py` | Average number of explanations per image |
| 2 | `analysis/compute_time.py` | Compute time per image (s/img) |
| 3 | `analysis/individual_coverage.py` | Individual coverage table + decay curves |
| 4 | `analysis/max_coverage.py` | Maximum coverage at K (greedy set cover) |
| 5 | `analysis/size_vs_coverage.py` | Explanation size vs individual coverage |
| 6 | `analysis/generalizability.py` | Generalizability at K (IoU across 50/50 splits) |
| 7 | `analysis/validity_ratio.py` | Validity ratio (LLM-graded plausibility) |
| 8 | `analysis/rcic.py` | Relative cumulative individual coverage at K |

All scripts share the same interface:

```bash
python analysis/<script>.py \
    --model resnet --dataset rival10 --vocab-size 300 --behavior B26 \
    --intermediate-results-dir intermediate_results
```
