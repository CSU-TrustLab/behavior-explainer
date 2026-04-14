import io
import os
import pickle
from pathlib import Path

import torch

# Default storage directory: behavior-explainer/intermediate_results/
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "intermediate_results"


class Pickler:
    def write(name, something, base_dir=None):
        base = Path(base_dir) if base_dir else _DEFAULT_DIR
        base.mkdir(parents=True, exist_ok=True)
        with open(base / f"{name}.pkl", "wb") as f:
            pickle.dump(something, f)

    def read(name, base_dir=None):
        base = Path(base_dir) if base_dir else _DEFAULT_DIR
        filename = base / f"{name}.pkl"
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except RuntimeError as e:
            if "Attempting to deserialize object on CUDA device" in str(e):
                print("Caught RuntimeError while loading. Remapping tensors to CPU.")
                with open(filename, "rb") as f:
                    return CPU_Unpickler(f).load()
            raise

    def create_or_read(name, f_create, base_dir=None):
        base = Path(base_dir) if base_dir else _DEFAULT_DIR
        if (base / f"{name}.pkl").is_file():
            return Pickler.read(name, base_dir)
        something = f_create()
        Pickler.write(name, something, base_dir)
        return something


class CPU_Unpickler(pickle.Unpickler):
    """Unpickler that remaps CUDA tensors to CPU on load.

    Reference: https://stackoverflow.com/questions/74259296
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)
