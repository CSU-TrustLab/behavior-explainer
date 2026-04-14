"""
concept_eraser.py — Abstract base class and concrete implementations for concept erasure.

Three erasure strategies are provided:
  - ClipOrthoEraser   : orthogonal projection erasure in CLIP space
  - ClipSpliceEraser  : SPLICE/ADMM-based sparse decomposition erasure
  - LeaceEraserWrapper: LEACE (Least-squares Concept Erasure) wrapper
                        https://github.com/EleutherAI/concept-erasure
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from concept_erasure import LeaceEraser  # https://github.com/EleutherAI/concept-erasure
from sklearn.linear_model import LogisticRegression
from torch import linalg as LA

_INTERMEDIATE_DIR = Path(__file__).resolve().parent.parent / "intermediate_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_device_dtype(x, device=None, dtype=None):
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else "cuda:2"
    if dtype is None:
        dtype = torch.float32
    return device, dtype


class ADMM:
    """ADMM solver for sparse decomposition (used by ClipSpliceEraser)."""

    def __init__(self, rho=1., l1_penalty=0.2, tol=1e-6, max_iter=10000, device="cuda", verbose=False):
        self.rho = rho
        self.l1_penalty = l1_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.verbose = verbose

    def step(self, Cb, Q_cho, z, u):
        xn = torch.cholesky_solve(2 * Cb + self.rho * (z - u), Q_cho)
        zn = torch.where((xn + u - self.l1_penalty / self.rho) > 0, xn + u - self.l1_penalty / self.rho, 0)
        un = u + xn - zn
        return xn, zn, un

    def fit(self, C, v):
        c = C.shape[0]
        Q = 2 * C @ C.T + (torch.eye(c) * self.rho).to(self.device)
        Q_cho = torch.linalg.cholesky(Q)
        x = torch.randn((c, v.shape[0])).to(self.device)
        z = torch.randn((c, v.shape[0])).to(self.device)
        u = torch.randn((c, v.shape[0])).to(self.device)
        for ix in range(self.max_iter):
            z_old = z
            x, z, u = self.step(C @ v.T, Q_cho, z, u)
            res_prim = torch.linalg.norm(x - z, dim=0)
            res_dual = torch.linalg.norm(self.rho * (z - z_old), dim=0)
            if (res_prim.max() < self.tol) and (res_dual.max() < self.tol):
                break
        if self.verbose:
            print("Stopping at iteration {}".format(ix))
            print("Prime Residual, r_k: {}".format(res_prim.mean()))
            print("Dual Residual, s_k: {}".format(res_dual.mean()))
        return z.T


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ConceptEraser(ABC):
    @abstractmethod
    def erase_some(self):
        pass

    @abstractmethod
    def erase_single(self):
        pass

    @abstractmethod
    def erase_all(self):
        pass


# ---------------------------------------------------------------------------
# ClipOrthoEraser
# ---------------------------------------------------------------------------

class ClipOrthoEraser(ConceptEraser):
    def __init__(self, C: torch.Tensor, device=None, dtype=None, rcond=1e-6):
        """
        C: (d, n) concept matrix (columns are c1..cn).
        Precomputes G^+ = pinv(C^T C) on the target device for reuse.
        """
        assert C.ndim == 2
        d, n = C.shape
        assert d == 512
        device, dtype = _as_device_dtype(C, device, dtype)
        self.C = C.to(device=device, dtype=dtype, non_blocking=True)
        G = self.C.T @ self.C
        self.Gpinv = torch.linalg.pinv(G, rcond=rcond)
        self.d, self.n = d, n
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def strength_matrix(self, e: torch.Tensor, C_names, behavior_id):
        assert len(C_names) == self.n
        e = e.to(device=self.device, dtype=self.dtype, non_blocking=True)
        strengths = e @ self.C
        df = pd.DataFrame(strengths.cpu().numpy(), columns=C_names)
        _INTERMEDIATE_DIR.mkdir(exist_ok=True)
        df.round(decimals=4).to_csv(
            _INTERMEDIATE_DIR / f"strengths_{behavior_id}_ortho.csv", index=False
        )

    @torch.no_grad()
    def erase_some(self, e: torch.Tensor, idxs) -> torch.Tensor:
        """
        Erase concepts indexed by idxs. Preserves all other C^T dot-products;
        minimal L2 change.

        e: (..., d) single vector or batch
        """
        if len(idxs) == 0:
            return e
        e = e.to(device=self.device, dtype=self.dtype, non_blocking=True)
        v = e @ self.C          # (..., n)
        mask = torch.zeros_like(v)
        mask[..., idxs] = 1.0
        Pv = mask * v           # (..., n)
        inner = Pv @ self.Gpinv.T
        delta = inner @ self.C.T
        return e - delta

    @torch.no_grad()
    def erase_single(self, e: torch.Tensor, i: int) -> torch.Tensor:
        return self.erase_some(e, [i])

    @torch.no_grad()
    def erase_all(self, e: torch.Tensor) -> torch.Tensor:
        return self.erase_some(e, list(range(self.n)))

    def to(self, device=None, dtype=None):
        """Move precomputed state to a new device/dtype."""
        if device is not None:
            self.C = self.C.to(device=device)
            self.Gpinv = self.Gpinv.to(device=device)
            self.device = device
        if dtype is not None:
            self.C = self.C.to(dtype=dtype)
            self.Gpinv = self.Gpinv.to(dtype=dtype)
            self.dtype = dtype
        return self


# ---------------------------------------------------------------------------
# ClipSpliceEraser
# ---------------------------------------------------------------------------

class ClipSpliceEraser(ConceptEraser):
    def __init__(self, C: torch.Tensor, e: torch.Tensor, device=None, dtype=None):
        """
        C: (d, n) concept matrix (columns are c1..cn).
        e: (..., d) single vector or batch.
        """
        assert C.ndim == 2
        d, n = C.shape
        assert d == 512
        device, dtype = _as_device_dtype(C, device, dtype)
        self.d, self.n = d, n
        self.device, self.dtype = device, dtype

        self.dictionary = C.to(device=device, dtype=dtype, non_blocking=True).T
        self.embedding = e.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if self.embedding.ndim == 1:
            self.embedding = torch.unsqueeze(self.embedding, 0)
        admm = ADMM(rho=5, tol=1e-6, max_iter=25, l1_penalty=0.01, device=self.device, verbose=False)
        self.weights = admm.fit(self.dictionary, self.embedding).to(self.device)

    @torch.no_grad()
    def strength_matrix(self, C_names, behavior_id):
        assert len(C_names) == self.n
        df = pd.DataFrame(self.weights.cpu().numpy(), columns=C_names)
        _INTERMEDIATE_DIR.mkdir(exist_ok=True)
        df.round(decimals=4).to_csv(
            _INTERMEDIATE_DIR / f"strengths_{behavior_id}_splice.csv", index=False
        )

    @torch.no_grad()
    def erase_some(self, idxs, verbose=False) -> torch.Tensor:
        """Erase concepts indexed by idxs. Returns reconstructed embedding."""
        mask = torch.ones_like(self.weights)
        mask[..., idxs] = 0.0
        selected_weights = mask * self.weights
        if verbose:
            print("+selected_weights.shape", selected_weights.shape)
        result = selected_weights @ self.dictionary
        if verbose:
            print("+result.shape", result.shape)
        if result.shape[0] == 1:
            result = torch.squeeze(result)
        return result

    @torch.no_grad()
    def erase_single(self, i: int) -> torch.Tensor:
        return self.erase_some([i])

    @torch.no_grad()
    def erase_all(self) -> torch.Tensor:
        return self.erase_some(list(range(self.n)))


# ---------------------------------------------------------------------------
# LeaceEraserWrapper
# LEACE (Least-squares Concept Erasure): https://github.com/EleutherAI/concept-erasure
# ---------------------------------------------------------------------------

class LeaceEraserWrapper(ConceptEraser):
    def __init__(self, X_train, C, device=None, dtype=None, keep_hashmap=False):
        assert C.ndim == 2
        d, n = C.shape
        assert d == 512
        device, dtype = _as_device_dtype(C, device, dtype)
        self.C = C.to(device=device, dtype=dtype, non_blocking=True)
        self.d, self.n = d, n
        self.device, self.dtype = device, dtype

        self.X_train = X_train.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.Z_train = X_train @ self.C  # concept strengths

        self.keep_hashmap = keep_hashmap
        self.hashmap = {}

    @torch.no_grad()
    def erase_some(self, some_X, idxs) -> torch.Tensor:
        idxs.sort()
        if self.keep_hashmap:
            idxs_s = ','.join([str(e) for e in idxs])
            if idxs_s not in self.hashmap:
                idxs_t = torch.LongTensor(idxs)
                eraser = LeaceEraser.fit(self.X_train, self.Z_train[:, idxs_t])
                self.hashmap[idxs_s] = eraser
                if len(self.hashmap) % 2000 == 0:
                    print("Hashmap len: {}. CUDA usage: {}%".format(
                        len(self.hashmap), torch.cuda.utilization(0)
                    ))
            return self.hashmap[idxs_s](some_X)
        else:
            idxs_t = torch.LongTensor(idxs)
            eraser = LeaceEraser.fit(self.X_train, self.Z_train[:, idxs_t])
            return eraser(some_X)

    @torch.no_grad()
    def erase_single(self, some_X, i: int) -> torch.Tensor:
        eraser = LeaceEraser.fit(self.X_train, self.Z_train[:, i])
        return eraser(some_X)

    @torch.no_grad()
    def erase_all(self, some_X) -> torch.Tensor:
        eraser = LeaceEraser.fit(self.X_train, self.Z_train)
        return eraser(some_X)

    @torch.no_grad()
    def strength_matrix(self, some_X, C_names, behavior_id):
        assert len(C_names) == self.n
        strengths = some_X @ self.C
        self.last_strengths = strengths
        df = pd.DataFrame(strengths.cpu().numpy(), columns=C_names)
        _INTERMEDIATE_DIR.mkdir(exist_ok=True)
        df.round(decimals=4).to_csv(
            _INTERMEDIATE_DIR / f"strengths_{behavior_id}_leace.csv", index=False
        )

    @torch.no_grad()
    def lr_coef_norm(self, some_X):
        X = some_X.cpu().numpy()
        Y = self.Z_train.cpu().numpy()
        lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs').fit(X, Y)
        beta = torch.from_numpy(lr.coef_)
        return beta.norm(p=torch.inf)

    @torch.no_grad()
    def cov_mat(self, some_X):
        some_Y = some_X @ self.C
        combined = torch.cat((some_X, some_Y), dim=1)
        return LA.matrix_norm(torch.cov(combined), ord=torch.inf)
