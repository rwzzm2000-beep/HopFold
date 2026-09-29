"""Small, dependency-light core of HopFold.

This public reference keeps the algorithmic contract visible without exposing
training orchestration, GPU kernels, preprocessing pipelines, or full runs.
Each feature channel stores one 8-bit code. The codebook stores two values for
that code: the reconstructed self feature and a precomputed first-hop
neighbor statistic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Carrier:
    codes: np.ndarray       # [num_nodes, feature_dim], uint8
    codebook: np.ndarray    # [feature_dim, 256, 2], float32


def _quantize(values: np.ndarray, levels: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel uniform quantization used only for the toy reference."""
    values = np.asarray(values, dtype=np.float32)
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    scale = np.maximum(hi - lo, 1e-8)
    codes = np.rint((values - lo) / scale * (levels - 1)).clip(0, levels - 1).astype(np.uint8)
    centers = np.linspace(0.0, 1.0, levels, dtype=np.float32)[None, :] * scale[:, None] + lo[:, None]
    return codes, centers


def aggregate_neighbors(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute a mean neighbor statistic for directed ``src -> dst`` edges."""
    x = np.asarray(x, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    out = np.zeros_like(x)
    degree = np.zeros(x.shape[0], dtype=np.int64)
    for src, dst in edges:
        out[dst] += x[src]
        degree[dst] += 1
    if np.any(degree == 0):
        raise ValueError("every destination node needs at least one neighbor")
    return out / degree[:, None]


def build_carrier(x: np.ndarray, edges: np.ndarray) -> Carrier:
    """Build a paired self/neighbor carrier from a small graph."""
    x = np.asarray(x, dtype=np.float32)
    neighbor_mean = aggregate_neighbors(x, edges)
    codes, self_centers = _quantize(x)
    # Pair the same self-feature code with the neighbor statistic observed on
    # the same nodes. This is the key idea: one code addresses two values.
    neighbor_centers = np.empty_like(self_centers)
    for channel in range(x.shape[1]):
        for code in range(256):
            mask = codes[:, channel] == code
            neighbor_centers[channel, code] = (
                neighbor_mean[mask, channel].mean() if np.any(mask) else 0.0
            )
    codebook = np.stack([self_centers, neighbor_centers], axis=-1).astype(np.float32)
    return Carrier(codes=codes, codebook=codebook)


def decode_carrier(carrier: Carrier) -> tuple[np.ndarray, np.ndarray]:
    """Decode self features and the stored first-hop statistic."""
    codes = np.asarray(carrier.codes, dtype=np.uint8)
    book = np.asarray(carrier.codebook, dtype=np.float32)
    if book.shape[:2] != (codes.shape[1], 256) or book.shape[2] != 2:
        raise ValueError("expected codebook shape [feature_dim, 256, 2]")
    channels = np.arange(codes.shape[1])[None, :]
    return book[channels, codes, 0], book[channels, codes, 1]


def hopfold_linear(carrier: Carrier, self_weight: np.ndarray, neighbor_weight: np.ndarray) -> np.ndarray:
    """Apply the paired carrier to a first-layer linear projection."""
    self_x, neighbor_x = decode_carrier(carrier)
    return self_x @ np.asarray(self_weight).T + neighbor_x @ np.asarray(neighbor_weight).T
