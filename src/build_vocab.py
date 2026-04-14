"""
build_vocab.py — Generate concept vocabularies for RIVAL10 and EuroSAT.

Loads a CLIP model from intermediate_results/clip.pkl, builds a vocabulary
of concepts by filtering MS-COCO words through Core WordNet and a cosine
similarity threshold, and saves the resulting word lists to vocabs/.

Usage:
    python src/build_vocab.py
"""

import sys
from pathlib import Path

import clip
import torch
from tqdm import tqdm

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pickler import Pickler

# ---------------------------------------------------------------------------
# Dataset class definitions
# ---------------------------------------------------------------------------

RIVAL10_CLASSES = ["truck", "car", "plane", "ship", "cat", "dog", "equine", "deer", "frog", "bird"]

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_prompts(path):
    with open(path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"+ Prompts loaded (n={len(prompts)})")
    return prompts


def load_mscoco(path):
    words = []
    with open(path, "r") as f:
        for line in f:
            words.append(line.strip())
    words.reverse()
    return words


def load_core_wordnet(path):
    cwn = {"a": set(), "n": set(), "v": set()}
    with open(path, "r") as f:
        for line in f:
            if len(line) > 0:
                pos = line[0]
                word = line[3: line.index("%")]
                cwn[pos].add(word)
    return cwn

# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def get_class_vectors(classes, clip_model, prompts):
    return clip_model.get_concept_vectors(classes, prompts, mean_centered=False, BMP=0)


def find_text_mean(vocab, clip_model, device):
    reps = []
    for word in tqdm(vocab, desc="Computing text mean"):
        tokens = clip.tokenize([word]).to(device)
        rep = clip_model.encode_text(tokens)
        reps.append(rep)
    reps = torch.stack(reps)
    return torch.mean(reps, dim=0).T


def get_concept_vocab(order_preference, N, candidates, avoid, avoid_vecs, clip_model, prompts, max_cos_sim):
    """Build a vocab of N concepts from candidates, filtered by cosine similarity."""
    vocab = []
    vocab_vecs = None
    for word in order_preference:
        if word not in candidates or word in avoid:
            continue
        word_vec = clip_model.get_concept_vectors([word], prompts, mean_centered=False, BMP=0)
        sims = word_vec.T @ avoid_vecs
        most_sim_idx = torch.argmax(sims, dim=1).item()
        if sims[0][most_sim_idx] >= max_cos_sim:
            print(f"Discard (too similar to class): {word} ~ {avoid[most_sim_idx]} = {sims[0][most_sim_idx]:.2f}")
            continue
        if vocab_vecs is None:
            vocab.append(word)
            vocab_vecs = word_vec
        else:
            sims = word_vec.T @ vocab_vecs
            most_sim_idx = torch.argmax(sims, dim=1).item()
            if sims[0][most_sim_idx] >= max_cos_sim:
                print(f"Discard (too similar to concept): {word} ~ {vocab[most_sim_idx]} = {sims[0][most_sim_idx]:.2f}")
            else:
                vocab.append(word)
                vocab_vecs = torch.cat((vocab_vecs, word_vec), dim=1)
        if len(vocab) >= N:
            break
    return vocab, vocab_vecs

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_vocab(name, vocab, vocab_vecs, class_vecs=None):
    out_dir = PROJECT_ROOT / "vocabs"
    out_dir.mkdir(exist_ok=True)

    txt_path = out_dir / f"{name}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(vocab))
    print(f"+ Saved vocab ({len(vocab)} words) to {txt_path}")

    Pickler.write(f"{name}_vecs", vocab_vecs)
    if class_vecs is not None:
        Pickler.write(f"{name}_class_vecs", class_vecs)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0"

    # Load CLIP model from intermediate results
    # Expected path: intermediate_results/clip.pkl
    print("Loading CLIP model...")
    text_to_concept_obj = Pickler.read("clip")
    text_to_concept_obj.device = device
    clip_model = text_to_concept_obj.clip_model.to(device)
    clip_model.device = device
    print("+ CLIP model loaded.")

    # Load reference data
    vocabs_dir = PROJECT_ROOT / "vocabs"
    prompts = load_prompts(vocabs_dir / "prompts.txt")
    mscoco = load_mscoco(vocabs_dir / "mscoco.txt")
    cwn = load_core_wordnet(vocabs_dir / "core-wordnet.txt")

    N = 2200
    max_cos_sim = 90
    candidates = cwn["n"] | cwn["a"]

    with torch.no_grad():
        text_mean = find_text_mean(mscoco, clip_model, device)

        for dataset_name, classes in [("RIVAL10", RIVAL10_CLASSES), ("EuroSAT", EUROSAT_CLASSES)]:
            print(f"\n--- Building vocab for {dataset_name} ---")
            class_vecs = get_class_vectors(classes, clip_model, prompts)

            vocab, vocab_vecs = get_concept_vocab(
                mscoco, N, candidates, classes, class_vecs, clip_model, prompts, max_cos_sim
            )
            print(f"Vocab size: {len(vocab)}, vec shape: {vocab_vecs.shape}")

            class_vecs = class_vecs - text_mean
            vocab_vecs = vocab_vecs - text_mean

            save_vocab(f"MCS_{max_cos_sim}_NA_{dataset_name}", vocab, vocab_vecs.cpu(), class_vecs.cpu())

    print("\nDone!")


if __name__ == "__main__":
    main()
