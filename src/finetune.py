"""
finetune.py — Fine-tune vision models on RIVAL10 and EuroSAT.

Four model × dataset combinations, all following the same format:
  - ResNet18 × EuroSAT  : SSL4EO self-supervised backbone (MoCo on Sentinel-2),
                          only the linear head is trained (linear probe).
  - ResNet18 × RIVAL10  : ImageNet pretrained, full fine-tuning.
  - VGG19    × RIVAL10  : ImageNet pretrained, full fine-tuning.
  - VGG19    × EuroSAT  : ImageNet pretrained, full fine-tuning.

Fine-tuned models are saved to intermediate_results/ via Pickler.

Usage:
    python src/finetune.py
"""

import sys
import warnings
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torchvision import models

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pickler import Pickler
from src.datasets import get_dataloader_eurosat, get_dataloader_rival10


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def get_resnet_eurosat():
    """
    ResNet18 for EuroSAT.
    Backbone is initialised from SSL4EO pretrained weights (loaded separately via
    load_ssl4eo_weights). Only the linear head is trainable — linear probing.
    """
    net = models.resnet18(pretrained=False)
    net.fc = torch.nn.Linear(512, 10)
    for name, param in net.named_parameters():
        if name not in ("fc.weight", "fc.bias"):
            param.requires_grad = False
    net.fc.weight.data.normal_(mean=0.0, std=0.01)
    net.fc.bias.data.zero_()
    return net


def get_resnet_rival10():
    """
    ResNet18 for RIVAL10.
    ImageNet pretrained backbone, full fine-tuning.
    """
    net = models.resnet18(pretrained=True)
    net.fc = torch.nn.Linear(512, 10)
    net.fc.weight.data.normal_(mean=0.0, std=0.01)
    net.fc.bias.data.zero_()
    return net


def get_vgg_rival10():
    """
    VGG19 for RIVAL10.
    ImageNet pretrained backbone, full fine-tuning.
    """
    net = models.vgg19(pretrained=True)
    net.classifier[6] = torch.nn.Linear(4096, 10)
    net.classifier[6].weight.data.normal_(mean=0.0, std=0.01)
    net.classifier[6].bias.data.zero_()
    return net


def get_vgg_eurosat():
    """
    VGG19 for EuroSAT.
    ImageNet pretrained backbone, full fine-tuning.
    """
    net = models.vgg19(pretrained=True)
    net.classifier[6] = torch.nn.Linear(4096, 10)
    net.classifier[6].weight.data.normal_(mean=0.0, std=0.01)
    net.classifier[6].bias.data.zero_()
    return net


# ---------------------------------------------------------------------------
# SSL4EO checkpoint loading  (ResNet × EuroSAT only)
# ---------------------------------------------------------------------------

def load_ssl4eo_weights(model, checkpoint_path):
    """
    Load MoCo SSL4EO pretrained weights into a ResNet, stripping the projection
    head so only the backbone is initialised. The fc head stays randomly
    initialised and will be trained from scratch.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        print(f"=> No SSL4EO checkpoint at '{checkpoint_path}'. Skipping.")
        return model

    print(f"=> Loading SSL4EO checkpoint '{checkpoint_path}'")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    cleaned = {
        k[len("module.encoder_q."):]: v
        for k, v in state_dict.items()
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc")
    }
    msg = model.load_state_dict(cleaned, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, msg.missing_keys
    print(f"=> SSL4EO weights loaded.")
    return model


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, epochs=5, lr=0.05):
    """Fine-tune model using SGD with momentum. Returns model on CPU."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.cuda()

    for epoch in range(epochs):
        model.train()
        running_loss = running_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            if i % 5 == 4:
                preds = torch.argmax(torch.sigmoid(outputs).detach().cpu(), dim=1)
                running_acc += accuracy_score(labels.cpu(), preds) * 100.0
            running_loss += loss.item()

            if i % 50 == 49:
                print(f"[{epoch+1}, {i+1:5d}]  loss: {running_loss/50:.3f}  acc: {running_acc/10:.3f}")
                running_loss = running_acc = 0.0

    return model.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # SSL4EO checkpoint — download separately and place at this path
    SSL4EO_CKPT = PROJECT_ROOT.parent / "SSL4EO_finetuned_weights" / "B3_rn18_moco_0099_ckpt.pth"

    configs = [
        {
            "name":   "resnet_eurosat",
            "model":  get_resnet_eurosat(),
            "loader": get_dataloader_eurosat(),
            "epochs": 5,
            "lr":     0.05,
            "pre_fn": lambda m: load_ssl4eo_weights(m, SSL4EO_CKPT),
        },
        {
            "name":   "resnet_rival10",
            "model":  get_resnet_rival10(),
            "loader": get_dataloader_rival10(train=True),
            "epochs": 5,
            "lr":     0.05,
            "pre_fn": None,
        },
        {
            "name":   "vgg_rival10",
            "model":  get_vgg_rival10(),
            "loader": get_dataloader_rival10(train=True),
            "epochs": 5,
            "lr":     0.05,
            "pre_fn": None,
        },
        {
            "name":   "vgg_eurosat",
            "model":  get_vgg_eurosat(),
            "loader": get_dataloader_eurosat(),
            "epochs": 5,
            "lr":     0.05,
            "pre_fn": None,
        },
    ]

    for cfg in configs:
        print(f"\n{'='*60}\nFine-tuning: {cfg['name']}\n{'='*60}")
        net = cfg["model"]
        if cfg["pre_fn"] is not None:
            net = cfg["pre_fn"](net)
        net = train(net, cfg["loader"], epochs=cfg["epochs"], lr=cfg["lr"])
        net.eval()
        Pickler.write(f"{cfg['name']}_iter1", net)
        print(f"Saved: {cfg['name']}_iter1")

    print("\nAll models fine-tuned and saved.")


if __name__ == "__main__":
    main()
