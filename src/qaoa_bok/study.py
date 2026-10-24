"""Experiment drivers returning pandas DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .qaoa import (cut_values, expected_best_of_k, hill_climb_best, noisy_probs,
                   random_regular_graph, statevector_probs, success_prob_best_of_k, train_noiseless)


def run_sweep(n: int, seeds: list[int], depths: list[int], eps2s: list[float], ks: list[int],
              ls_reps: int = 200) -> pd.DataFrame:
    """For each graph seed, depth, noise level and budget K record best-of-K metrics.

    Parameters are trained noiselessly (the usual pre-hardware workflow) and deployed under noise.
    Methods: qaoa, qaoa_mean (expected cut, K=0), random (uniform sampling), hillclimb (same K evaluations).
    """
    rows = []
    for s in seeds:
        g = random_regular_graph(n, s)
        cuts = cut_values(g)
        cmax = cuts.max()
        unif = np.full(2**n, 2.0**-n)
        rng = np.random.default_rng(1000 + s)
        hc = {k: np.mean([hill_climb_best(cuts, n, k, rng) for _ in range(ls_reps)]) for k in ks}
        for k in ks:
            rows.append(dict(n=n, seed=s, p=0, eps2=np.nan, K=k, method="random",
                             ratio=expected_best_of_k(unif, cuts, k) / cmax,
                             success=success_prob_best_of_k(unif, cuts, k)))
            rows.append(dict(n=n, seed=s, p=0, eps2=np.nan, K=k, method="hillclimb",
                             ratio=hc[k] / cmax, success=np.nan))
        for p in depths:
            params = train_noiseless(cuts, n, p, seed=s)
            ideal = statevector_probs(cuts, n, params)
            for e in eps2s:
                pr = ideal if e == 0 else noisy_probs(g, cuts, params, e, e / 10)
                for k in ks:
                    rows.append(dict(n=n, seed=s, p=p, eps2=e, K=k, method="qaoa",
                                     ratio=expected_best_of_k(pr, cuts, k) / cmax,
                                     success=success_prob_best_of_k(pr, cuts, k)))
                rows.append(dict(n=n, seed=s, p=p, eps2=e, K=0, method="qaoa_mean",
                                 ratio=float(pr @ cuts) / cmax, success=np.nan))
    return pd.DataFrame(rows)


def crossover_eps(df: pd.DataFrame, p: int, k: int) -> dict[int, float]:
    """Per graph seed, smallest swept eps2 at which QAOA best-of-K no longer beats random sampling."""
    out: dict[int, float] = {}
    rnd = df[(df.method == "random") & (df.K == k)].set_index("seed").ratio
    q = df[(df.method == "qaoa") & (df.K == k) & (df.p == p)]
    for s, sub in q.groupby("seed"):
        sub = sub.sort_values("eps2")
        bad = sub[sub.ratio <= rnd[s]]
        out[int(s)] = float(bad.eps2.iloc[0]) if len(bad) else float("inf")
    return out
