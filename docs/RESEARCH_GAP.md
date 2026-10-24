# Research gap

## State of the art (from the supplied literature)
- QAOA is a leading variational heuristic for combinatorial optimisation; the review by Blekos et al., 2024 (doi:10.1016/j.physrep.2024.03.002) stresses hardware-specific challenges such as error susceptibility and noise resilience.
- Small Qiskit MaxCut studies (Al Salti & Asif, 2027, doi:10.1007/978-3-032-32862-5_1) report that QAOA finds all optimal bitstrings on a 3-node graph and improves with depth, but on noiseless, tiny instances.
- Kim et al., 2023 (doi:10.1038/s41586-023-06096-3) argue noise is the main impediment before fault tolerance; Cai et al., 2023 (doi:10.1103/RevModPhys.95.045005) survey mitigation and list open problems for mitigation-based devices.
- Wang et al., 2027 (doi:10.15302/frontphys.2027.023301) say the long-term viability of VQAs needs systematic reassessment.

## Gap
QAOA is almost always scored by its *expected* cut (approximation ratio). An optimiser, however, consumes *samples* and keeps the best one. Noise hurts the expectation, but the best of K samples depends on the tail of the output distribution, which is a different (and for optimisation more relevant) quantity. It is rarely stated (a) how quickly the best-of-K advantage over plain random sampling and over a classical local search with the *same number of cut evaluations* degrades with gate noise, and (b) at which noise level adding a QAOA layer stops paying off when the criterion is best-of-K rather than the mean.

## Hypotheses (falsifiable)
- **H1** At a fixed sample budget K, noiseless-trained QAOA (p=1,2) on 3-regular MaxCut with n=8 keeps a best-of-K approximation ratio above uniform random sampling for two-qubit depolarising error up to eps2 = 0.1. Falsified if the QAOA mean falls to the random level (within one std over 5 graphs) for any eps2 <= 0.1 at K in {5,10,30,100}.
- **H2** The depth p=2 advantage over p=1 reverses at a lower noise level for the mean cut than for best-of-K. Success metric: location of the p=1/p=2 crossing in eps2 for each metric.

Scale: exact density-matrix simulation, n = 6-10 qubits, depolarising noise only. No claim of advantage is made.
