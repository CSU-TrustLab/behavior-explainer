# Tests

## Prerequisites

Complete pipeline steps 1–4 before running any integration test.  All
intermediate artefacts must be present in `behavior-explainer/intermediate_results/`:

| File | Produced by |
|------|-------------|
| `clip.pkl` | step 0 (legacy codebase) |
| `resnet_rival10_finetuned.pkl` | step 1 — `src/finetune.py` |
| `resnet_rival10_to_clip.pkl` | step 2 — `src/train_aligner.py` |
| `clip_to_resnet_rival10.pkl` | step 2 — `src/train_aligner.py` |
| `MCS_90_NA_RIVAL10_vecs.pkl` | step 3 — `src/build_vocab.py` |
| `MCS_90_NA_RIVAL10_class_vecs.pkl` | step 3 — `src/build_vocab.py` |

And in `behavior-explainer/vocabs/`:

| File | Produced by |
|------|-------------|
| `MCS_90_NA_RIVAL10.txt` | step 3 — `src/build_vocab.py` |

And the RIVAL10 dataset split file must exist at:

```
behavior-explainer/datasets/RIVAL10/meta/train_test_split_by_url.json
```

If any of these files are missing the integration tests are automatically
skipped (they are guarded by `@pytest.mark.skipif`).

---

## Running the tests

Install all dependencies from the project root, then run pytest:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

```bash
python -m pytest tests/ -v -s
```

To run a single test:

```bash
python -m pytest tests/test_run_experiment.py::test_run_experiment_all_configs_rival10_planes -v -s
```

---

## Test catalogue

### Unit tests (no GPU / no intermediate files required)

These always run and validate the binary CSV format helpers independently of
any model or data.

| Test | What it checks |
|------|---------------|
| `test_parse_binary_csv_basic` | Two-block file is parsed into correct blocks and tuples |
| `test_parse_binary_csv_empty_block` | A blank line between two blocks produces an empty middle block |
| `test_assert_csv_invariants_valid` | A well-formed CSV passes all invariant checks |
| `test_assert_csv_invariants_size_mismatch` | Raises when `size ≠ popcount(pos) + popcount(neg)` |
| `test_assert_csv_invariants_mutual_exclusion` | Raises when concept appears as both positive and negative |
| `test_assert_csv_invariants_wrong_length` | Raises when bit-string length ≠ `n_concepts` |
| `test_write_binary_csv_round_trip` | `_write_binary_csv` output survives a full parse-and-check round trip |
| `test_write_binary_csv_cross_image_signs` | A concept may carry opposite signs across different images |

### Integration tests (require steps 1–4 + GPU)

#### `test_class_distribution_correct` / `test_class_distribution_misclassified`

Informational diagnostics that load the behavior pkl and print per-class
instance counts.  Use them to decide which class to pass as `--class-idx`
before running the all-configs test.

`test_class_distribution_correct` counts correctly-classified instances
(behavior B=2: `predicted == label == class_idx`).

`test_class_distribution_misclassified` counts misclassified instances
(behavior B=3: `predicted ≠ class_idx`, `label == class_idx`).

Both tests are skipped if the behavior pkl has not been built yet (run
`test_run_experiment_all_configs_rival10_ships` first to trigger a build).
They pass as long as at least one class has ≥ `_MAX_IMAGES` qualifying instances.

#### `test_run_experiment_xpenum_rival10_cars`

Runs **XpEnum only** (XpSatEnum and NaiveEnum disabled) for all three erasers
(Ortho, Splice, LEACE) on RIVAL10 **behavior 2** (correctly classified cars,
`class_idx=1`).

Key parameters:

| Parameter | Value |
|-----------|-------|
| Vocabulary size (`--n-concepts`) | 300 |
| Max shared images | 50 |
| NINA iterations | 50 |
| Per-configuration timeout | 300 s |
| Results written to | `intermediate_results/` (temporary, via `tmp_path`) |

The test passes if Ortho+XpEnum completes and produces at least one non-empty
AXp explanation.  LEACE is expected to time out; that is not a failure.

#### `test_run_experiment_all_configs_rival10_ships`

Runs **all 9 configurations** — (XpEnum + XpSatEnum + NaiveEnum) ×
(Ortho + Splice + LEACE) — on RIVAL10 **behavior 2** (correctly classified
ships, `class_idx=3`).

Key parameters:

| Parameter | Value |
|-----------|-------|
| Vocabulary size (`--n-concepts`) | 300 |
| Max shared images | 50 |
| NINA iterations | 50 |
| Per-configuration timeout | 300 s |
| Results written to | `intermediate_results/CM_MCS90_N300_resnet_rival10_e/` (persistent) |

**Shared-instance selection:** before running any configuration, the pipeline
screens all candidate bird images and keeps only those that produce at least
one non-empty AXp or CXp for **all three erasers simultaneously**.  This
guarantees every configuration operates on the same image set, making results
directly comparable.  If fewer than 50 images pass the criterion the user is
prompted to retry with the available count (`--auto-retry` bypasses the
prompt in CI / automated contexts).

NaiveEnum always times out at depth 2 within 300 s; that is expected and not
a failure.  The test asserts that `ortho+XpEnum` completes and that the total
AXp count across completed configurations is > 0.

**Output files** (written to `intermediate_results/CM_MCS90_N300_resnet_rival10_e/`):

```
binary_COXB23-N300_resnet_rival10_A.csv   # Ortho  + XpEnum     — AXps
binary_COXB23-N300_resnet_rival10_C.csv   # Ortho  + XpEnum     — CXps
binary_COSB23-N300_resnet_rival10_A.csv   # Ortho  + XpSatEnum  — AXps
binary_COSB23-N300_resnet_rival10_C.csv   # Ortho  + XpSatEnum  — CXps
binary_CSXB23-N300_resnet_rival10_A.csv   # Splice + XpEnum     — AXps
...
time_COXB23-N300_resnet_rival10.csv       # wall-clock time for each config
```

These CSV files are **not deleted after the test** and are used downstream for
metric calculation.

---

## Binary CSV format

Each output file stores signed concept explanations in a compact binary
encoding.  One blank-line-separated block per image; one row per explanation:

```
<size>,<positive_bits>,<negative_bits>
```

- `size` — number of concepts in the explanation (= `popcount(pos) + popcount(neg)`)
- `positive_bits` — binary string of length `n_concepts`; bit `i` is 1 if concept `i`
  appears with a positive sign (positive cosine similarity / projection)
- `negative_bits` — binary string of length `n_concepts`; bit `i` is 1 if concept `i`
  appears with a negative sign

Invariants enforced by `assert_csv_invariants`:
1. Bit strings have exactly `n_concepts` characters.
2. `size == popcount(positive_bits) + popcount(negative_bits)`.
3. No concept has both bits set (`pos[i] == neg[i] == 1` is forbidden).
4. A concept **may** carry different signs across different images.

Sign conventions by eraser:
- **Ortho**: sign of the dot product of the image embedding and the concept vector.
- **Splice**: all signs are positive (per the SPLICE paper's definition).
- **LEACE**: sign of the cosine similarity between the image embedding and the
  concept vector.