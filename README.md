# Behavior Explainer

Code for the paper: **[Title]**

## Repository Structure

```
behavior-explainer/
├── src/                    # Core source code
│   ├── datasets.py         # RIVAL10 and EuroSAT dataset classes and dataloaders
│   ├── finetune.py         # Fine-tuning ResNet18 and VGG19 on RIVAL10 / EuroSAT
│   ├── train_aligner.py    # Penultimate-layer extraction and CLIP linear aligner training
│   └── build_vocab.py      # Concept vocabulary generation via CLIP
├── utils/                  # Shared utilities
│   ├── pickler.py          # Compressed binary serialisation (Pickler / CPU_Unpickler)
│   ├── concept_eraser.py   # Concept erasure: ClipOrthoEraser, ClipSpliceEraser, LeaceEraserWrapper
│   └── mhs.py              # Minimal hitting set solvers (exact, random, hybrid)
├── datasets/               # Image datasets (not tracked — see per-folder READMEs for download links)
│   ├── RIVAL10/
│   └── EuroSAT/
├── vocabs/                 # Concept vocabularies and reference word lists
├── intermediate_results/   # Pickled tensors and trained models (not tracked)
├── scripts/                # Entry-point scripts for reproducing experiments
├── notebooks/              # Exploratory notebooks
├── results/                # Evaluation outputs and metrics
└── tests/                  # Unit tests
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

### 2. Build concept vocabularies

```bash
python src/build_vocab.py
```

Generates concept word lists for each dataset and saves them to `vocabs/`.

### 3. Train CLIP linear aligners

```bash
python src/train_aligner.py
```

Extracts penultimate-layer representations from each fine-tuned model, trains bidirectional
linear maps to/from CLIP space, and reports round-trip reconstruction quality.
