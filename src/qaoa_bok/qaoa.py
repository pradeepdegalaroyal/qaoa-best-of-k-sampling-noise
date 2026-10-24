"""MaxCut QAOA: noiseless statevector and noisy density-matrix simulation."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.optimize import minimize

_I2 = np.eye(2)
_DELTA4 = np.einsum("ik,jl->ijkl", _I2, _I2) / 4.0


def random_regular_graph(n: int, seed: int, d: int = 3) -> nx.Graph:
    """Random d-regular graph on n nodes."""
    return nx.random_regular_graph(d, n, seed=seed)


def cut_values(g: nx.Graph) -> np.ndarray:
    """Cut value of every bitstring (index = integer, qubit 0 = most significant bit)."""
    n = g.number_of_nodes()
    bits = ((np.arange(2**n)[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)
    c = np.zeros(2**n)
    for u, v in g.edges:
        c += bits[:, u] ^ bits[:, v]
    return c


def _mixer_state(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    u = np.array([[c, s], [s, c]])
    t = psi.reshape((2,) * n)
    for q in range(n):
        t = np.moveaxis(np.tensordot(u, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def statevector_probs(cuts: np.ndarray, n: int, params: np.ndarray) -> np.ndarray:
    """Output distribution of ideal QAOA with params = [gamma_1..gamma_p, beta_1..beta_p]."""
    p = len(params) // 2
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for k in range(p):
        psi = psi * np.exp(-1j * params[k] * cuts)
        psi = _mixer_state(psi, params[p + k], n)
    return np.abs(psi) ** 2


def _depolarise(rho: np.ndarray, qubits: tuple[int, ...], lam: float, n: int) -> np.ndarray:
    """Depolarising channel (1-lam) rho + lam * I/d (x) tr_qubits(rho) on 1 or 2 qubits."""
    if lam <= 0:
        return rho
    k = len(qubits)
    src = list(qubits) + [n + q for q in qubits]
    t = np.moveaxis(rho, src, list(range(2 * k)))
    if k == 1:
        red = np.einsum("ii...->...", t)
        mixed = np.einsum("ik,...->ik...", _I2 / 2.0, red)
    else:
        red = np.einsum("ijij...->...", t)
        mixed = np.einsum("ijkl,...->ijkl...", _DELTA4, red)
    out = (1 - lam) * t + lam * mixed
    return np.moveaxis(out, list(range(2 * k)), src)


def noisy_probs(g: nx.Graph, cuts: np.ndarray, params: np.ndarray, eps2: float, eps1: float) -> np.ndarray:
    """Output distribution under depolarising noise.

    Each RZZ edge gate is followed by a two-qubit depolarising channel of strength ``eps2``
    and each RX mixer rotation by a single-qubit one of strength ``eps1``. The diagonal cost
    layer is applied as one block, then the per-edge channels (this only reorders commuting terms).
    """
    n = g.number_of_nodes()
    p = len(params) // 2
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    rho = np.outer(psi, psi.conj()).reshape((2,) * (2 * n))
    for k in range(p):
        ph = np.exp(-1j * params[k] * cuts)
        rho = (ph[:, None] * rho.reshape(2**n, 2**n) * ph.conj()[None, :]).reshape((2,) * (2 * n))
        for u, v in g.edges:
            rho = _depolarise(rho, (u, v), eps2, n)
        b = params[p + k]
        c, s = np.cos(b), -1j * np.sin(b)
        u_m = np.array([[c, s], [s, c]])
        for q in range(n):
            rho = np.moveaxis(np.tensordot(u_m, rho, axes=([1], [q])), 0, q)
            rho = np.moveaxis(np.tensordot(u_m.conj(), rho, axes=([1], [n + q])), 0, n + q)
            rho = _depolarise(rho, (q,), eps1, n)
    d = np.clip(np.real(np.diagonal(rho.reshape(2**n, 2**n))), 0, None)
    return d / d.sum()


def train_noiseless(cuts: np.ndarray, n: int, p: int, seed: int, restarts: int = 6) -> np.ndarray:
    """Maximise the ideal expected cut with multi-start L-BFGS-B."""
    rng = np.random.default_rng(seed)
    best, best_val = None, -np.inf
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 1.0, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(lambda x: -statevector_probs(cuts, n, x) @ cuts, x0, method="L-BFGS-B")
        if -res.fun > best_val:
            best, best_val = res.x, -res.fun
    assert best is not None
    return best


def expected_best_of_k(probs: np.ndarray, cuts: np.ndarray, k: int) -> float:
    """Exact E[max cut among k i.i.d. samples] from an output distribution."""
    vals = np.unique(cuts)
    cdf = np.clip(np.array([probs[cuts <= v].sum() for v in vals]), 0, 1)
    prev = np.concatenate([[0.0], cdf[:-1]])
    return float(np.sum(vals * (cdf**k - prev**k)))


def success_prob_best_of_k(probs: np.ndarray, cuts: np.ndarray, k: int) -> float:
    """Probability that at least one of k samples is a maximum cut."""
    return float(1 - (1 - probs[cuts == cuts.max()].sum()) ** k)


def hill_climb_best(cuts: np.ndarray, n: int, budget: int, rng: np.random.Generator) -> float:
    """Stochastic 1-flip hill climbing with a budget of cut evaluations (random restart if stuck)."""
    x = int(rng.integers(2**n))
    cx = cuts[x]
    used, stuck, best = 1, 0, cx
    while used < budget:
        y = x ^ (1 << int(rng.integers(n)))
        cy = cuts[y]
        used += 1
        if cy > cx:
            x, cx, stuck = y, cy, 0
        else:
            stuck += 1
        best = max(best, cx)
        if stuck >= 2 * n and used < budget:
            x = int(rng.integers(2**n))
            cx = cuts[x]
            used += 1
            stuck = 0
            best = max(best, cx)
    return float(best)
