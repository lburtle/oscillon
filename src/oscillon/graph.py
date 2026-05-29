# src/oscillon/graph.py
# Graph construction and utilities script

# Used for deliberately constructing discrete graph structures

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class MotifSpec:
    """Specification for a single motif subgraph"""

    n: int  # number of neurons
    adjacency: np.ndarray  # (n,n) bool; A[i,j]=True means j->i
    label: str = ""
    eps_init: float = 0.1
    delta_init: float = 0.5

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"Motif({self.n})"

    @classmethod
    def cyclic(cls, n: int, label: str = "", **kwargs: float) -> "MotifSpec":
        """Create a cycle on n nodes"""
        A = np.zeros((n, n), dtype=bool)
        for j in range(n):
            A[(j + 1) % n, j] = True
        return cls(n=n, adjacency=A, label=label or f"Cycle({n})", **kwargs)

    @classmethod
    def from_edge_list(
        cls, n: int, edges: list[tuple[int, int]], label: str = "", **kwargs: float
    ) -> "MotifSpec":
        """
        Create from explicit edge list
        edges: list of (source, target) tuples; source -> target
        Stored as A[target, source] = True
        """
        A = np.zeros((n, n), dtype=bool)
        for src, tgt in edges:
            A[tgt, src] = True
        return cls(n=n, adjacency=A, label=label or f"Custom({n})", **kwargs)


@dataclass
class NetworkSpec:
    """
    Full network specification: list of motifs plus inter-motif coupling
    """

    motifs: list[MotifSpec]
    delta_cross: float = 0.05
    # Per-pair cross coupling
    cross_deltas: np.ndarray | None = None
    # cross-deltas[i][j] = uniform coupling from motif j to motif i

    def __post_init__(self) -> None:
        k = len(self.motifs)
        if self.cross_deltas is None:
            self.cross_deltas = np.full((k, k), self.delta_cross)
            # Ensure diagonal is zero (no self-cross coupling)
            np.fill_diagonal(self.cross_deltas, 0.0)

    @property
    def _cross_deltas(self) -> NDArray[np.float64]:
        assert self.cross_deltas is not None, "cross_deltas not initialized"
        return self.cross_deltas

    @property
    def n_total(self) -> int:
        return sum(motif.n for motif in self.motifs)

    @property
    def k(self) -> int:
        return len(self.motifs)

    @property
    # Define which nodes belong to which motif (slices of entire node list)
    def slices(self) -> list[slice]:
        starts = np.concatenate(
            ([0], np.cumsum([motif.n for motif in self.motifs])[:-1])
        ).astype(int)
        return [
            slice(start, start + motif.n)
            for start, motif in zip(starts, self.motifs, strict=True)
        ]

    @property
    def labels(self) -> list[str]:
        return [motif.label for motif in self.motifs]

    def to_numpy_W(self) -> np.ndarray:
        """
        Build the full numpy weight matrix using each motif's
        eps_init and delta_init
        """
        N = self.n_total
        W = np.zeros((N, N))
        slices = self.slices

        for i, (slice_i, motif_i) in enumerate(zip(slices, self.motifs, strict=True)):
            for j, (slice_j, motif_j) in enumerate(
                zip(slices, self.motifs, strict=True)
            ):
                if i == j:
                    # Within-motif block
                    A = motif_i.adjacency.astype(float)
                    block = np.where(
                        A, -1.0 + motif_i.eps_init, -1.0 - motif_i.delta_init
                    ) * (1 - np.eye(motif_i.n))
                    W[slice_i, slice_j] = block
                else:
                    # Cross-motif block
                    assert self.cross_deltas is not None
                    d_cross = self.cross_deltas[i, j]
                    W[slice_i, slice_j] = np.full(
                        (motif_i.n, motif_j.n), -1.0 - d_cross
                    )
        return W

    def to_torch_adjacency(self) -> torch.Tensor:
        """
        Build full (N, N) boolean adjacency tensor
        Cross-motif connection are all False (non-edges)
        """
        N = self.n_total
        A = torch.zeros(N, N, dtype=torch.bool)
        for slice, motif in zip(self.slices, self.motifs, strict=True):
            A[slice, slice] = torch.from_numpy(motif.adjacency)
        return A

    def summary(self) -> None:
        logger.info(
            "NetworkSpec: %d motifs, %d n_total neurons total", self.k, self.n_total
        )
        for i, motif in enumerate(self.motifs):
            n_edges = motif.adjacency.sum()
            logger.info(
                "[%d] %s: %d neurons, %d edges, eps=%g, delta=%g",
                i,
                motif.label,
                motif.n,
                n_edges,
                motif.eps_init,
                motif.delta_init,
            )
        logger.info("delta_cross matrix:\n%s ", np.round(self._cross_deltas, 3))
