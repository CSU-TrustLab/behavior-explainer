"""
train_aligner.py — Extract penultimate-layer representations and train linear
                   aligners between a vision model's embedding space and CLIP.

Subtasks:
  B) Feature extractors — expose the penultimate-layer (intermediate) vector
     and the final linear head, treating the model as a white box consistent
     with the linear representation hypothesis.
  C) Representation extraction — paired tensors: one from the vision model
     feature extractor, one from CLIP's image encoder.
  D) Linear aligner: vision model space → CLIP space.
  E) Linear aligner: CLIP space → vision model space.
  F) Round-trip evaluation — compare original embeddings to
     clip_to_model(model_to_clip(e_model)) using MSE and R².

All aligners are saved to intermediate_results/ via Pickler.

Usage:
    python src/train_aligner.py
"""

import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pickler import Pickler
from src.datasets import get_dataloader_eurosat, get_dataloader_rival10


# ---------------------------------------------------------------------------
# B) Feature extractors
# ---------------------------------------------------------------------------

def get_feature_extractor(net, model_type):
    """
    Return a feature extractor for the penultimate layer of the model.

    The model is treated as two parts:
      - feature_extractor : everything up to (but not including) the final
                            linear classifier → produces the 'intermediate' vector
      - head              : the final linear layer → produces class logits

    This decomposition is consistent with the linear representation hypothesis:
    the intermediate vector is the representation studied for concept activations,
    and the head is a linear readout on top of it.

    Args:
        net        : fine-tuned model (CPU)
        model_type : 'resnet' or 'vgg'

    Returns:
        (feature_extractor, head, intermediate_dim)
          feature_extractor : callable (B, 3, H, W) → (B, intermediate_dim)
          head              : nn.Module
          intermediate_dim  : int (512 for ResNet, 4096 for VGG)
    """
    if model_type == "resnet":
        backbone = torch.nn.Sequential(*list(net.children())[:-1])

        def feature_extractor(x):
            return backbone(x).squeeze(dim=(2, 3))  # (B, 512)

        return feature_extractor, net.fc, 512

    elif model_type == "vgg":
        # Stage 1: conv features + adaptive avg pool (everything before classifier)
        fe_stage1 = torch.nn.Sequential(*list(net.children())[:-1])
        # Stage 2: classifier without the final head layer
        fe_stage2 = torch.nn.Sequential(*list(net.classifier.children())[:-1])

        def feature_extractor(x):
            x = fe_stage1(x)
            x = torch.flatten(x, 1)
            return fe_stage2(x)  # (B, 4096)

        return feature_extractor, net.classifier[-1], 4096

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'resnet' or 'vgg'.")


# ---------------------------------------------------------------------------
# C) Representation extraction
# ---------------------------------------------------------------------------

def extract_representations(net, clip_enc, dataloader, model_type, device):
    """
    Collect paired penultimate-layer representations from the vision model
    and CLIP over the same images.

    Args:
        net        : fine-tuned vision model
        clip_enc   : CLIP's input_to_representation callable
        dataloader : yields (images, labels) batches
        model_type : 'resnet' or 'vgg'
        device     : e.g. 'cuda:0'

    Returns:
        (reps_model, reps_clip) — CPU tensors, shapes (N, dim_model) and (N, 512)
    """
    feature_extractor, _, _ = get_feature_extractor(net, model_type)
    net.to(device).eval()

    reps_model, reps_clip = [], []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            reps_model.append(feature_extractor(inputs))
            reps_clip.append(clip_enc(inputs).float())

    reps_model = torch.vstack(reps_model).cpu()
    reps_clip  = torch.vstack(reps_clip).cpu()
    print(f"  reps_model: {reps_model.shape},  reps_clip: {reps_clip.shape}")
    return reps_model, reps_clip


# ---------------------------------------------------------------------------
# D & E) Linear aligner
# ---------------------------------------------------------------------------

class VectorToVectorRegression(torch.nn.Module):
    """Single linear layer mapping between two embedding spaces."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def _r2(target, pred):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return (1 - ss_res / ss_tot).item()


def train_aligner(source_train, target_train, source_test, target_test,
                  device, epochs=50000, lr=0.01, log_every=5000):
    """
    Train a linear map from source space to target space using SGD + MSE loss.

    Dimensions are inferred from the data, so this works for both
    model→CLIP (e.g. 512→512 or 4096→512) and CLIP→model directions.

    Returns the trained aligner on CPU.
    """
    in_dim, out_dim = source_train.shape[1], target_train.shape[1]
    aligner = VectorToVectorRegression(in_dim, out_dim).to(device)

    s_tr = source_train.to(device)
    t_tr = target_train.to(device)
    s_te = source_test.to(device)
    t_te = target_test.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(aligner.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs + 1):
        aligner.train()
        pred = aligner(s_tr)
        loss = criterion(pred, t_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            aligner.eval()
            with torch.no_grad():
                test_loss = criterion(aligner(s_te), t_te)
            print(f"  [{epoch}/{epochs}]  train={loss.item():.4f}  test={test_loss.item():.4f}")

    return aligner.cpu()


# ---------------------------------------------------------------------------
# F) Round-trip evaluation
# ---------------------------------------------------------------------------

def evaluate_roundtrip(model_to_clip, clip_to_model, reps_model_train, reps_model_test):
    """
    Evaluate the information preserved by the round-trip:
        e_model  →  model_to_clip  →  clip_to_model  →  e_model_hat

    Reports MSE and R² for both train and test splits.
    """
    criterion = torch.nn.MSELoss()
    print("Round-trip: e_model vs. clip_to_model(model_to_clip(e_model))")
    for split, e_model in [("Train", reps_model_train), ("Test", reps_model_test)]:
        e_model = e_model.cpu()
        with torch.no_grad():
            e_hat = clip_to_model(model_to_clip(e_model))
        mse = criterion(e_model, e_hat).item()
        r2  = _r2(e_model, e_hat)
        print(f"  {split}: MSE={mse:.4f},  R²={r2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0"

    print("Loading CLIP model...")
    text_to_concept_obj = Pickler.read("clip")
    text_to_concept_obj.device = device
    clip_model = text_to_concept_obj.clip_model.to(device)
    clip_model.device = device
    clip_enc = clip_model.input_to_representation
    print("+ CLIP loaded.")

    configs = [
        {"name": "resnet_eurosat", "model_type": "resnet", "loader": get_dataloader_eurosat()},
        {"name": "resnet_rival10", "model_type": "resnet", "loader": get_dataloader_rival10(train=True)},
        {"name": "vgg_rival10",    "model_type": "vgg",    "loader": get_dataloader_rival10(train=True)},
        {"name": "vgg_eurosat",    "model_type": "vgg",    "loader": get_dataloader_eurosat()},
    ]

    for cfg in configs:
        name, model_type, loader = cfg["name"], cfg["model_type"], cfg["loader"]
        print(f"\n{'='*60}\nAligner: {name}\n{'='*60}")

        # B) Load fine-tuned model; feature_extractor + head is the white-box decomposition
        net = Pickler.read(f"{name}_iter1")
        net.eval()

        # C) Extract paired representations
        reps_model, reps_clip = extract_representations(net, clip_enc, loader, model_type, device)
        model_train, model_test, clip_train, clip_test = train_test_split(
            reps_model, reps_clip, test_size=0.2, random_state=42
        )

        # D) model → CLIP
        print(f"\n[D] Training {name} → CLIP")
        model_to_clip = train_aligner(
            model_train, clip_train, model_test, clip_test,
            device=device, epochs=50000, lr=0.01, log_every=5000,
        )
        Pickler.write(f"{name}_to_clip", model_to_clip)
        print(f"Saved: {name}_to_clip")

        # E) CLIP → model
        print(f"\n[E] Training CLIP → {name}")
        clip_to_model = train_aligner(
            clip_train, model_train, clip_test, model_test,
            device=device, epochs=10000, lr=0.01, log_every=2500,
        )
        Pickler.write(f"clip_to_{name}", clip_to_model)
        print(f"Saved: clip_to_{name}")

        # F) Round-trip evaluation
        print(f"\n[F] Round-trip evaluation: {name}")
        evaluate_roundtrip(model_to_clip, clip_to_model, model_train, model_test)

    print("\nDone!")


if __name__ == "__main__":
    main()
