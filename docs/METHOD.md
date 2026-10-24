# Method

- **Problem:** MaxCut on random 3-regular graphs, n = 8 (main sweep, graph seeds 0-4), n = 6, 8, 10 (scaling, seeds 0-2). Optimum by brute force.
- **Quantum method:** standard QAOA (Hadamard start, cost layer, RX mixer), depth p = 1, 2. Parameters are trained once on the ideal statevector (multi-start L-BFGS-B, 6 starts) and then deployed unchanged under noise, the usual pre-hardware workflow.
- **Noise:** exact density-matrix simulation. A two-qubit depolarising channel of strength eps2 follows every RZZ edge gate and a single-qubit one of strength eps2/10 follows every RX. eps2 in {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4}. No readout error, no coherent error, no transpilation (all-to-all connectivity); each gate is one channel, not a CNOT decomposition.
- **Metric:** the output distribution is exact, so E[best cut in K i.i.d. samples]/max cut is computed in closed form from its CDF (no sampling error); K in {5, 10, 30, 100}. Also P(at least one optimum in K) and the mean cut ratio.
- **Baselines:** (1) uniform random sampling, K samples (the noise floor of QAOA). (2) Classical: stochastic 1-flip hill climbing with random restarts and the same budget of K cut evaluations, 200 repetitions per graph. The standard quantum baseline is noiseless QAOA at the same p (eps2 = 0).
- **Ablation:** depth p = 1 vs 2; mean-cut objective vs best-of-K as the evaluation criterion.
- **Robustness sweep:** eps2 above, plus scaling in n at eps2 in {0, 0.01, 0.05}, K = 30.
- **Seeds:** 5 graphs (main), 3 graphs (scaling); mean +- std across graphs.
