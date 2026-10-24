"""Reproduce all results: python experiments/run_all.py [--quick]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qaoa_bok.study import crossover_eps, run_sweep

ROOT = pathlib.Path(__file__).resolve().parents[1]
EPS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
KS = [5, 10, 30, 100]
K_SCALE = 30
EPS_SCALE = [0.0, 0.01, 0.05]


def agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.groupby(by, dropna=False).ratio.agg(["mean", "std"]).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    res, fig = ROOT / "results", ROOT / "figures"
    res.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)
    seeds = [0, 1] if a.quick else [0, 1, 2, 3, 4]
    eps = [0.0, 0.01, 0.4] if a.quick else EPS
    n_main = 6 if a.quick else 8
    ns = [6] if a.quick else [6, 8, 10]
    reps = 30 if a.quick else 200

    main_df = run_sweep(n_main, seeds, [1, 2], eps, KS, ls_reps=reps)
    main_df.to_csv(res / f"sweep_n{n_main}.csv", index=False)
    scale_df = pd.concat([run_sweep(n, seeds[:3], [1, 2], EPS_SCALE, [K_SCALE], ls_reps=reps) for n in ns])
    scale_df.to_csv(res / "scaling.csv", index=False)

    summary: dict = {"n_main": n_main, "seeds": seeds, "crossover_eps2_per_seed": {}}
    for p in (1, 2):
        for k in KS:
            summary["crossover_eps2_per_seed"][f"p{p}_K{k}"] = {
                str(s): v for s, v in crossover_eps(main_df, p, k).items()}
    tab = agg(main_df[main_df.method.isin(["qaoa", "qaoa_mean"])], ["method", "p", "eps2", "K"])
    tab.to_csv(res / "summary_qaoa.csv", index=False)
    base = agg(main_df[main_df.method.isin(["random", "hillclimb"])], ["method", "K"])
    base.to_csv(res / "summary_baselines.csv", index=False)
    suc = (main_df[main_df.method.isin(["qaoa", "random"])]
           .groupby(["method", "p", "eps2", "K"], dropna=False).success.agg(["mean", "std"]).reset_index())
    suc.to_csv(res / "summary_success.csv", index=False)

    f, axs = plt.subplots(1, len(KS), figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axs, KS):
        for p, col in ((1, "C0"), (2, "C3")):
            t = tab[(tab.method == "qaoa") & (tab.p == p) & (tab.K == k)]
            xs = np.where(t.eps2 == 0, 1e-3, t.eps2)
            ax.errorbar(xs, t["mean"], t["std"], color=col, marker="o", capsize=3, label=f"QAOA p={p}")
        for m, col, ls in (("random", "k", "--"), ("hillclimb", "C2", "-.")):
            v = base[(base.method == m) & (base.K == k)].iloc[0]
            ax.axhline(v["mean"], color=col, ls=ls, label=m + " (same K)")
        ax.set_xscale("log")
        ax.set_title(f"K = {k} samples")
        ax.set_xlabel("2-qubit depolarising error $\\epsilon_2$ (0 plotted at 1e-3)")
    axs[0].set_ylabel(f"E[best cut] / max cut (n={n_main})")
    axs[0].legend(fontsize=7)
    f.tight_layout()
    f.savefig(fig / "best_of_k_vs_noise.png", dpi=200)
    plt.close(f)

    sc = agg(scale_df, ["method", "p", "eps2", "n"])
    f, ax = plt.subplots(figsize=(5.8, 4.2))
    for p, col in ((1, "C0"), (2, "C3")):
        for e, mk in zip(EPS_SCALE, "osv"):
            t = sc[(sc.method == "qaoa") & (sc.p == p) & (sc.eps2 == e)]
            ax.errorbar(t.n, t["mean"], t["std"], color=col, marker=mk, capsize=3,
                        ls="-" if e == 0 else ":", label=f"p={p}, $\\epsilon_2$={e}")
    for m, col in (("random", "k"), ("hillclimb", "C2")):
        t = sc[sc.method == m]
        ax.errorbar(t.n, t["mean"], t["std"], color=col, ls="--", marker="x", capsize=3, label=m)
    ax.set_xlabel("qubits n")
    ax.set_ylabel("E[best cut] / max cut, K=30")
    ax.legend(fontsize=6)
    f.tight_layout()
    f.savefig(fig / "scaling_n.png", dpi=200)
    plt.close(f)

    summary["runtime_s"] = round(time.time() - t0, 1)
    (res / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done in", summary["runtime_s"], "s")


if __name__ == "__main__":
    main()
